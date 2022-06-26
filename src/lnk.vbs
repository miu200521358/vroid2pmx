Option Explicit

Dim fs
Set fs = WScript.CreateObject("WScript.Shell")

Dim file_path
file_path = WScript.Arguments(0)

Dim cmd
cmd = WScript.Arguments(1)

Dim arg
arg = WScript.Arguments(2)
Dim rep_arg
rep_arg = Replace(arg, "\", Chr(34))

Dim icon_loc
icon_loc = WScript.Arguments(3)

WScript.Echo("-----------------------")
WScript.Echo(file_path)
WScript.Echo(cmd)
WScript.Echo(arg)
WScript.Echo(rep_arg)
WScript.Echo("-----------------------")

Dim fn
Set fn = fs.CreateShortcut(file_path)
fn.TargetPath = cmd
fn.Arguments = rep_arg
fn.IconLocation = icon_loc
fn.save()

WScript.Quit(1)
