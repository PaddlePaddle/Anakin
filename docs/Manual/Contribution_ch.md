# 如何贡献代码

我们真诚地感谢您的贡献，欢迎通过 GitHub 的 fork 和 pull request 流程来提交代码。

## Contributor License Agreements

在您的代码合入之前请签署个人或者公司的Contributor License Agreement(CLA)。

- 如果您个人是原始代码的拥有者，并拥有代码的知识产权，您需要签署[个人CLA](https://gist.github.com/tanzhongyibidu/6605bdef5f7bb03b9084dd8fed027037)    
- 如果原始代码属于公司，并且公司同意提交代码到我们的仓储，那您需要签署[公司CLA](https://gist.github.com/tanzhongyibidu/709c675c1e79804e3e871f8c1e62292d)    

请您选择合适的CLA并仔细阅读，在您签署CLA后方可将代码合入。

## 添加License

在新提交的代码中包含license：

- c++代码头文件

```c++
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/
```

- python代码

```python
# Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## 代码要求

- 代码注释请遵守[Doxygen](http://www.stack.nl/~dimitri/doxygen/)的样式
- 所有代码必须具有单元测试
- 通过所有单元测试
- 请遵守提交代码的一些约定

以下教程将指导您提交代码

## Fork
首先跳转到[Anakin](https://github.com/PaddlePaddle/Anakin)的github首页，然后点击`Fork`, 生成自己目录下的仓库

## 克隆（clone）

将远程仓库clone到本地：

```bash
git clone YOUR_REPOSITORY_URL
cd Anakin
```

## 创建本地分支
Anakin目前使用[Git流分支模型](https://nvie.com/posts/a-successful-git-branching-model/)进行开发, 测试和维护。  
所有的feature和bug fix的开发工作都应该在一个新的分支上完成，根据需要从现有分支上创建新分支。  
使用`git checkout -b`创建并切换到新分支
```bash
git checkout -b YOUR_NEW_BRANCH
```

## 开始开发

编写代码


## 构建和测试

详细请参考[Docker installation guide](docker/README.md) 和 [build from source guide](docs/Manual/INSTALL_en.md)。


## 提交（commit）

提交代码时，请认真写好提交说明，这样其他人就可以清楚的知道这次提交做了哪些改变：
```bash
git commit -m 'description'
```

## 保持本地仓库最新

在发起Pull Request之前，需要与原始仓库同步。

如果还没添加原仓库，请先添加源，可通过`git remote -v`查看是否添加源：
```bash
git remote -v
origin .... (fetch)
origin .... (push)
```
如果只出现origin，说明还未添加源，可通过如下命令添加源：
```bash
git remote add upstream ORIGIN_REPOSITORY_URL
```
获取 upstream 的最新代码并更新当前分支
```bash
git fetch upstream
git pull upstream BRANCH_NAME
```
## Push到远程仓库

将本地的修改push到远程仓库上
```bash
git push origin BRANCH_NAME
```

## 提交Pull Request

切换到所建分支，然后点击`New pull request`。  
![](./contri1.JPG)

选择目标分支：  
![](./contri2.JPG)

接下来等待review。

## 删除远程分支
在PR被merge进主仓库后，可以在PR的界面删除远程仓库的分支。  
也可以通过以下命令删除远程分支：
```bash
git push origin :YOUR_NEW_BRANCH
```

## 删除本地分支

最后，删除本地分支。
```bash
#切换到其他分支
git checkout OTHER_BRANCH

#删除YOUR_NEW_BRANCH分支
git branch -D YOUR_NEW_BRANCH
```

至此，我们就完成了一次代码贡献的过程。

## 提交代码的一些约定

为了使评审人在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

1. 提交Pull Request前：  
- 注意commit的数量

  - 原因：如果仅仅修改一个文件但提交了十几个commit，每个commit只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个commit才能知道做了哪些修改，且不排除commit之间的修改存在相互覆盖的情况。

  - 建议：每次提交时，保持尽量少的commit，可以通过`git commit --amend`补充上次的commit。对已经Push到远程仓库的多个commit，可以参考[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)
  
- 注意每个commit的名称：应能反映当前commit的内容，不能太随意。

2. 如果解决了某个Issue的问题，请在该Pull Request的第一个评论框中加上：`fix #issue_number`，这样当该Pull Request被合并后，会自动关闭对应的Issue。关键词包括：close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考[Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

在回复评审人意见时，请您遵守以下约定：  
1. 评审人的每个意见都必须回复
   - 对评审意见同意且按其修改完的，给个简单的Done即可
   - 对评审意见不同意的，请给出您自己的反驳理由。
2. 如果评审意见比较多
   - 请给出总体的修改情况。
   - 请采用[start a review](https://help.github.com/articles/reviewing-proposed-changes-in-a-pull-request/)进行回复，而非直接回复的方式。
