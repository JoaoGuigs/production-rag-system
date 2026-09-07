param([string]$MsgFile)
$lines = Get-Content $MsgFile -ErrorAction SilentlyContinue
if ($lines) {
    $lines | Where-Object { $_ -notmatch 'Co-authored-by: Cursor <cursoragent@cursor.com>' } | Set-Content $MsgFile -Encoding utf8
}
