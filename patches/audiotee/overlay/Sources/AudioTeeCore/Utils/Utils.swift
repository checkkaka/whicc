import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation

// MARK: - Audio Device Utilities

/// Checks if an audio device is valid and alive
func isAudioDeviceValid(_ deviceID: AudioObjectID) -> Bool {
  var address = getPropertyAddress(selector: kAudioDevicePropertyDeviceIsAlive)

  var isAlive: UInt32 = 0
  var size = UInt32(MemoryLayout<UInt32>.size)
  let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &isAlive)

  let valid = status == kAudioHardwareNoError && isAlive == 1

  AudioTeeLogging.logger.debug(
    "Checked device validity",
    context: [
      "device_id": String(deviceID),
      "status": String(status),
      "is_alive": String(isAlive),
      "valid": String(valid),
    ])
  return valid
}

/// Creates an AudioObjectPropertyAddress with the given selector and optional scope/element
func getPropertyAddress(
  selector: AudioObjectPropertySelector,
  scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
  element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
) -> AudioObjectPropertyAddress {
  return AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
}

/// Translates process IDs to AudioObjectIDs.
///
/// whicc patch: when `allowPartial` is true, PIDs that are not currently
/// producing audio (no Audio Object yet) are skipped instead of failing the
/// whole launch. If every PID fails, still throws so callers do not silently
/// fall back to tapping all system audio.
func translatePIDsToProcessObjects(_ pids: [Int32], allowPartial: Bool = false) throws
  -> [AudioObjectID]
{
  guard !pids.isEmpty else {
    return []
  }

  var processObjects: [AudioObjectID] = []
  var failedPIDs: [Int32] = []

  for pid in pids {
    var address = getPropertyAddress(selector: kAudioHardwarePropertyTranslatePIDToProcessObject)
    var processObject: AudioObjectID = 0
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var mutablePid = pid

    let status = AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject),
      &address,
      UInt32(MemoryLayout<pid_t>.size),
      &mutablePid,
      &size,
      &processObject
    )

    if status == kAudioHardwareNoError && processObject != kAudioObjectUnknown {
      processObjects.append(processObject)
      AudioTeeLogging.logger.debug(
        "Translated PID to process object",
        context: [
          "pid": String(pid),
          "process_object": String(processObject),
        ])
    } else {
      failedPIDs.append(pid)
      AudioTeeLogging.logger.debug(
        "Failed to translate PID to process object",
        context: [
          "pid": String(pid),
          "status": String(status),
        ])
    }
  }

  if !failedPIDs.isEmpty {
    if allowPartial && !processObjects.isEmpty {
      AudioTeeLogging.logger.info(
        "Skipped PIDs with no audio object",
        context: [
          "failed_pids": failedPIDs.map(String.init).joined(separator: ", "),
          "kept": String(processObjects.count),
        ])
      return processObjects
    }
    throw AudioTeeError.pidTranslationFailed(failedPIDs)
  }

  return processObjects
}
