Option Explicit

Dim scores() As Double
Dim count As Integer

Sub Main()
    ReDim scores(10)
    Dim i As Integer
    For i = 0 To 9
        scores(i) = Rnd() * 100
    Next i
    
    Dim avg As Double
    avg = ComputeAverage()
    
    If avg > 50 Then
        MsgBox "Above average: " & avg
    Else
        MsgBox "Below average: " & avg
    End If
    
    Select Case Int(avg / 10)
        Case 0 To 4
            Call HandleLow
        Case 5 To 7
            Call HandleMid
        Case Else
            Call HandleHigh
    End Select
End Sub

Function ComputeAverage() As Double
    On Error GoTo ErrHandler
    Dim total As Double
    Dim i As Integer
    For i = 0 To UBound(scores)
        total = total + scores(i)
    Next i
    ComputeAverage = total / (UBound(scores) + 1)
    Exit Function
ErrHandler:
    ComputeAverage = 0
End Function

Sub RiskyStuff()
    On Error Resume Next
    ReDim scores(100)
    GoTo cleanup
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
cleanup:
    Set fso = Nothing
End Sub
