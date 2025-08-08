# 在Windows环境中运行VALOR脚本

由于Windows环境中可能无法直接运行bash脚本，我们提供了等效的PowerShell脚本版本。这些脚本具有与原始bash脚本相同的功能，但可以在Windows PowerShell中直接运行。

## 安装依赖

在运行脚本之前，请确保已安装所有必要的Python依赖。我们提供了一个PowerShell脚本来帮助您安装这些依赖：

```powershell
# 安装所需的Python依赖
.\scripts\install_dependencies.ps1
```

## 使用方法

安装依赖后，在PowerShell终端中，可以通过以下方式运行脚本：

```powershell
# 运行测试脚本
.\scripts\test_valor.ps1
.\scripts\test_valor+.ps1
.\scripts\test_valor++.ps1

# 运行训练脚本
.\scripts\train_valor.ps1
.\scripts\train_valor+.ps1
.\scripts\train_valor++.ps1
```

## 脚本说明

这些PowerShell脚本是从原始bash脚本转换而来，保持了相同的参数和功能：

- `test_valor.ps1`：测试基础VALOR模型
- `test_valor+.ps1`：测试VALOR+模型
- `test_valor++.ps1`：测试VALOR++模型
- `train_valor.ps1`：训练基础VALOR模型
- `train_valor+.ps1`：训练VALOR+模型
- `train_valor++.ps1`：训练VALOR++模型

## 注意事项

1. 确保您的系统中已安装Python，并且可以通过命令行访问。
2. 确保所有必要的依赖项已安装（参见项目根目录的README.md）。
3. 如果遇到权限问题，可能需要调整PowerShell的执行策略：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
4. 这些脚本使用相对路径引用数据和模型目录，因此请确保从项目根目录运行脚本。