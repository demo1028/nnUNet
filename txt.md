conda activate nnunetv2

set NNUNET_PREPROCESSED=D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_preprocessed
set NNUNET_RAW=D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_raw
set NNUNET_RESULTS=D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_results

$env:NNUNET_PREPROCESSED = "D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_preprocessed"
$env:NNUNET_RAW = "D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_raw"
$env:NNUNET_RESULTS = "D:/DPP/graduate/Experiment/nnUNet/nnUNetFrame/nnUNet_results"

# ====== 验证是否设置成功 ======
echo $env:NNUNET_RAW  # PowerShell读取环境变量的语法