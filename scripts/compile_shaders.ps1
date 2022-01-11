Get-ChildItem -Path .\crates\examples\assets\shaders\ -File -Recurse -exclude *.spv | ForEach-Object { 
    $sourcePath = $_.fullname
    $targetPath = "$($_.fullname).spv"
    glslangValidator --target-env spirv1.4 -V -o $targetPath $sourcePath
}
