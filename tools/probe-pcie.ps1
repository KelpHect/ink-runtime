$gpu = Get-PnpDevice -Class DISPLAY | Where-Object { $_.FriendlyName -match 'Radeon' }
$gpu | Select-Object FriendlyName, InstanceId, Status | Format-List
try {
  $props = Get-PnpDeviceProperty -InstanceId $gpu.InstanceId -ErrorAction Stop
  $props | Where-Object { $_.KeyName -match 'Bus|Link|PCI' -or $_.Data -match 'PCI|x16|Gen' } | Select-Object KeyName, Data | Format-Table -AutoSize
} catch { Write-Output "prop enum failed: $_" }
