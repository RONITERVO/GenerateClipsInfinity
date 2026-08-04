Option Explicit
Dim shell, environment, fileSystem, appDir, pythonw, comfyRoot, app
Set shell = CreateObject("WScript.Shell")
Set environment = shell.Environment("PROCESS")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
appDir = fileSystem.GetParentFolderName(WScript.ScriptFullName)
pythonw = environment("WAN_PYTHONW_EXE")
If pythonw = "" Then
  comfyRoot = environment("WAN_COMFY_ROOT")
  If comfyRoot = "" Then comfyRoot = "D:\AI\ComfyUI"
  pythonw = fileSystem.BuildPath(comfyRoot, ".venv\Scripts\pythonw.exe")
End If
app = fileSystem.BuildPath(appDir, "app.py")
If Not fileSystem.FileExists(pythonw) Then
  MsgBox "Python was not found at " & pythonw & ". Set WAN_PYTHONW_EXE or WAN_COMFY_ROOT.", 16, "Wan Video Studio"
  WScript.Quit 1
End If
shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False
WScript.Sleep 1800
shell.Run "http://127.0.0.1:7868", 1, False
