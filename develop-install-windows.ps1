rm -Recurse ./build -ErrorAction SilentlyContinue

pip install -e . 

Get-ChildItem -Path ./build -Filter *.pyd -Recurse | ForEach-Object {
	Write-Host "Copying: $($_.FullName)"
	Copy-Item -Path $_.FullName -Destination ./traceon/backend
}
