Option Explicit
Dim shell, environment, fileSystem, appDir, pythonw, comfyRoot, app, hostUrl, i, http, ready

Set shell = CreateObject("WScript.Shell")
Set environment = shell.Environment("PROCESS")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

appDir = fileSystem.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = appDir

pythonw = environment("WAN_PYTHONW_EXE")
If pythonw = "" Then
  pythonw = environment("WAN_PYTHON_EXE")
End If

If pythonw = "" Then
  comfyRoot = environment("WAN_COMFY_ROOT")
  If comfyRoot = "" Then comfyRoot = "D:\AI\ComfyUI"
  
  If fileSystem.FileExists(fileSystem.BuildPath(comfyRoot, ".venv\Scripts\pythonw.exe")) Then
    pythonw = fileSystem.BuildPath(comfyRoot, ".venv\Scripts\pythonw.exe")
  ElseIf fileSystem.FileExists(fileSystem.BuildPath(comfyRoot, ".venv\Scripts\python.exe")) Then
    pythonw = fileSystem.BuildPath(comfyRoot, ".venv\Scripts\python.exe")
  End If
End If

app = fileSystem.BuildPath(appDir, "app.py")

If pythonw = "" Or Not fileSystem.FileExists(pythonw) Then
  MsgBox "Python was not found at: " & vbCrLf & pythonw & vbCrLf & vbCrLf & "Please set WAN_PYTHONW_EXE or WAN_COMFY_ROOT environment variable.", 16, "Wan Endless Theater"
  WScript.Quit 1
End If

shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False

hostUrl = "http://127.0.0.1:7868"
ready = False

On Error Resume Next
For i = 1 To 20
  WScript.Sleep 600
  Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
  If Err.Number <> 0 Then
    Set http = CreateObject("MSXML2.XMLHTTP")
  End If
  If Not http Is Nothing Then
    http.setTimeouts 500, 500, 500, 500
    http.Open "GET", hostUrl, False
    http.Send
    If Err.Number = 0 Then
      If http.Status = 200 Then
        ready = True
        Exit For
      End If
    End If
    Err.Clear
  End If
Next
On Error GoTo 0

shell.Run hostUrl, 1, False


