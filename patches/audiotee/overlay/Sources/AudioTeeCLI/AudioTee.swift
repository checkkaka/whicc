import AudioTeeCore
import CoreAudio
import Foundation

/// Exit code 2 = include-processes were requested but none currently have an
/// audio object (app not playing yet). Parents should retry without treating
/// this as a hard failure, and must NOT fall back to tapping all audio.
enum ExitCode: Error {
  case failure
  case waitingForAudioObjects
}

extension ExitCode {
  var code: Int32 {
    switch self {
    case .failure:
      return 1
    case .waitingForAudioObjects:
      return 2
    }
  }
}

struct AudioTee {
  var includeProcesses: [Int32] = []
  var excludeProcesses: [Int32] = []
  var mute: Bool = false
  var stereo: Bool = false
  var sampleRate: Double?
  var chunkDuration: Double = 0.2

  init() {}

  static func main() {
    let parser = SimpleArgumentParser(
      programName: "audiotee",
      abstract: "Capture system audio and stream to stdout",
      discussion: """
        AudioTee captures system audio using Core Audio taps and streams it as structured output.

        Process filtering:
        • include-processes: Only tap specified process IDs (empty = all processes)
        • exclude-processes: Tap all processes except specified ones
        • mute: How to handle processes being tapped

        whicc extensions:
        • Soft-skip PIDs that are not currently emitting audio (when at least one PID works)
        • Exit code 2 when include-processes is set but no PID has an audio object yet
        • Stdin NDJSON reconfigure: {"cmd":"set_include_processes","pids":[1,2,3]}
          rebuilds the tap in-process (avoids macOS 26 TCC kill+respawn issues)

        Examples:
          audiotee                              # Auto format, tap all processes
          audiotee --sample-rate 16000          # Convert to 16kHz mono for ASR
          audiotee --include-processes 1234 5678 9012
        """
    )

    parser.addArrayOption(
      name: "include-processes",
      help: "Process IDs to include (space-separated, empty = all processes)")
    parser.addArrayOption(
      name: "exclude-processes", help: "Process IDs to exclude (space-separated)")
    parser.addFlag(name: "mute", help: "Mute processes being tapped")
    parser.addFlag(name: "stereo", help: "Records in stereo")
    parser.addOption(
      name: "sample-rate",
      help: "Target sample rate (8000, 16000, 22050, 24000, 32000, 44100, 48000)")
    parser.addOption(
      name: "chunk-duration", help: "Audio chunk duration in seconds", defaultValue: "0.2")

    do {
      try parser.parse()

      var audioTee = AudioTee()
      audioTee.includeProcesses = try parser.getArrayValue("include-processes", as: Int32.self)
      audioTee.excludeProcesses = try parser.getArrayValue("exclude-processes", as: Int32.self)
      audioTee.mute = parser.getFlag("mute")
      audioTee.stereo = parser.getFlag("stereo")
      audioTee.sampleRate = try parser.getOptionalValue("sample-rate", as: Double.self)
      audioTee.chunkDuration = try parser.getValue("chunk-duration", as: Double.self)

      try audioTee.validate()
      try audioTee.run()

    } catch ArgumentParserError.helpRequested {
      parser.printHelp()
      exit(0)
    } catch ArgumentParserError.validationFailed(let message) {
      print("Error: \(message)", to: &standardError)
      exit(1)
    } catch let error as ArgumentParserError {
      print("Error: \(error.description)", to: &standardError)
      parser.printHelp()
      exit(1)
    } catch let code as ExitCode {
      exit(code.code)
    } catch {
      print("Error: \(error)", to: &standardError)
      exit(1)
    }
  }

  func validate() throws {
    if !includeProcesses.isEmpty && !excludeProcesses.isEmpty {
      throw ArgumentParserError.validationFailed(
        "Cannot specify both --include-processes and --exclude-processes")
    }
  }

  func run() throws {
    setupSignalHandlers()

    AudioTeeLogging.logger.info("Starting AudioTee...")

    guard chunkDuration > 0 && chunkDuration <= 5.0 else {
      AudioTeeLogging.logger.error(
        "Invalid chunk duration",
        context: ["chunk_duration": String(chunkDuration), "valid_range": "0.0 < duration <= 5.0"])
      throw ExitCode.failure
    }

    let runtime = AudioTeeRuntime(
      includeProcesses: includeProcesses,
      excludeProcesses: excludeProcesses,
      mute: mute,
      stereo: stereo,
      sampleRate: sampleRate,
      chunkDuration: chunkDuration
    )
    try runtime.start()
    runtime.startStdinListener()

    while true {
      let result = CFRunLoopRunInMode(CFRunLoopMode.defaultMode, 0.1, false)
      if result == CFRunLoopRunResult.stopped || result == CFRunLoopRunResult.finished {
        break
      }
    }

    AudioTeeLogging.logger.info("Shutting down...")
    runtime.stop()
  }

  private func setupSignalHandlers() {
    signal(SIGINT) { _ in
      AudioTeeLogging.logger.info("Received SIGINT, initiating graceful shutdown...")
      CFRunLoopStop(CFRunLoopGetMain())
    }
    signal(SIGTERM) { _ in
      AudioTeeLogging.logger.info("Received SIGTERM, initiating graceful shutdown...")
      CFRunLoopStop(CFRunLoopGetMain())
    }
  }
}

/// Long-lived capture state that can rebuild the tap without exiting the process.
final class AudioTeeRuntime {
  private var includeProcesses: [Int32]
  private let excludeProcesses: [Int32]
  private let mute: Bool
  private let stereo: Bool
  private let sampleRate: Double?
  private let chunkDuration: Double

  private let audioTapManager = AudioTapManager()
  private let outputHandler = BinaryAudioOutputHandler()
  private var recorder: AudioRecorder?
  private let lock = NSLock()
  private var stdinThread: Thread?

  init(
    includeProcesses: [Int32],
    excludeProcesses: [Int32],
    mute: Bool,
    stereo: Bool,
    sampleRate: Double?,
    chunkDuration: Double
  ) {
    self.includeProcesses = includeProcesses
    self.excludeProcesses = excludeProcesses
    self.mute = mute
    self.stereo = stereo
    self.sampleRate = sampleRate
    self.chunkDuration = chunkDuration
  }

  func start() throws {
    try applyConfiguration(includeProcesses: includeProcesses)
  }

  func stop() {
    lock.lock()
    defer { lock.unlock() }
    recorder?.stopRecording()
    recorder = nil
    audioTapManager.teardown()
  }

  func startStdinListener() {
    let thread = Thread { [weak self] in
      self?.readStdinLoop()
    }
    thread.name = "audiotee-stdin"
    thread.start()
    stdinThread = thread
  }

  private func readStdinLoop() {
    while let line = readLine(strippingNewline: true) {
      let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
      if trimmed.isEmpty { continue }
      guard let data = trimmed.data(using: .utf8),
        let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let cmd = obj["cmd"] as? String
      else {
        emitReconfigure(status: "error", detail: "invalid_json")
        continue
      }
      if cmd == "set_include_processes" {
        let raw = obj["pids"] as? [Any] ?? []
        let pids: [Int32] = raw.compactMap { value in
          if let i = value as? Int { return Int32(i) }
          if let n = value as? NSNumber { return n.int32Value }
          if let s = value as? String { return Int32(s) }
          return nil
        }
        // Reject empty include lists: reconfiguring to "no processes" must
        // never silently widen the tap to all system audio.
        if pids.isEmpty {
          emitReconfigure(status: "error", detail: "empty_pids_rejected")
          continue
        }
        DispatchQueue.main.async {
          do {
            try self.applyConfiguration(includeProcesses: pids)
            self.emitReconfigure(
              status: "ok",
              detail: pids.map(String.init).joined(separator: ","))
          } catch ExitCode.waitingForAudioObjects {
            // Stay alive with no active tap; parent will send another command.
            self.lock.lock()
            self.recorder?.stopRecording()
            self.recorder = nil
            self.audioTapManager.teardown()
            self.includeProcesses = pids
            self.lock.unlock()
            self.emitReconfigure(status: "waiting", detail: "no_valid_pids")
          } catch {
            self.emitReconfigure(status: "error", detail: String(describing: error))
          }
        }
      } else {
        emitReconfigure(status: "error", detail: "unknown_cmd")
      }
    }
  }

  private func emitReconfigure(status: String, detail: String) {
    let payload: [String: Any] = [
      "message_type": "reconfigure",
      "status": status,
      "detail": detail,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: payload),
      let line = String(data: data, encoding: .utf8)
    {
      print(line, to: &standardError)
      fflush(stderr)
    }
  }

  private func applyConfiguration(includeProcesses: [Int32]) throws {
    lock.lock()
    defer { lock.unlock() }

    self.includeProcesses = includeProcesses

    let (processes, isExclusive) = convertProcessFlags(
      include: includeProcesses, exclude: excludeProcesses)

    // Never silently widen to "all processes" when the parent asked for includes.
    if !includeProcesses.isEmpty && processes.isEmpty {
      throw ExitCode.waitingForAudioObjects
    }

    let tapConfig = TapConfiguration(
      processes: processes,
      muteBehavior: mute ? .muted : .unmuted,
      isExclusive: isExclusive,
      isMono: !stereo
    )

    recorder?.stopRecording()
    recorder = nil

    do {
      try audioTapManager.setupAudioTap(with: tapConfig)
    } catch AudioTeeError.pidTranslationFailed(let failedPIDs) {
      AudioTeeLogging.logger.error(
        "Failed to translate process IDs to audio objects",
        context: [
          "failed_pids": failedPIDs.map(String.init).joined(separator: ", "),
          "suggestion": "Target app may not be playing audio yet",
        ])
      audioTapManager.teardown()
      throw ExitCode.waitingForAudioObjects
    } catch {
      AudioTeeLogging.logger.error(
        "Failed to setup audio tap", context: ["error": String(describing: error)])
      throw ExitCode.failure
    }

    guard let deviceID = audioTapManager.getDeviceID() else {
      AudioTeeLogging.logger.error("Failed to get device ID from audio tap manager")
      throw ExitCode.failure
    }

    let newRecorder = try AudioRecorder(
      deviceID: deviceID, outputHandler: outputHandler, convertToSampleRate: sampleRate,
      chunkDuration: chunkDuration)
    try newRecorder.startRecording()
    recorder = newRecorder
  }

  private func convertProcessFlags(include: [Int32], exclude: [Int32]) -> ([Int32], Bool) {
    if !include.isEmpty {
      return (include, false)
    } else if !exclude.isEmpty {
      return (exclude, true)
    } else {
      return ([], true)
    }
  }
}

var standardError = FileHandle.standardError

extension FileHandle: TextOutputStream {
  public func write(_ string: String) {
    let data = Data(string.utf8)
    self.write(data)
  }
}
