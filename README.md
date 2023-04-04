# 基于语音和手势识别的有声相册

## 项目背景

“站在科技与人文的十字路口，科技产品应有更多的人文关怀”

——by Steven Jobs

我开发这个项目的初衷是希望通过语音交互等手段来让视障群体更真切地感受到这多彩的世界。视力正常的人即使由于疫情等原因不能去旅游，也可以通过浏览风景照片等方式领略自然风光。然而，视障群体只能看到一张张模糊的图片，无法同常人一样通过图片来感受自然的美好。好在大自然的元素是丰富的，一张图片背后往往有它的背景音效，例如视障群体虽然无法看清拍打在沙滩上的浪花，却能通过浪潮的起伏声来感受；虽然无法看清林间清澈的溪流，却能通过潺潺的流水声来感受。同样，通过图片背后的声音，视障群体也可以感受到朝露和晚霞，感受到三月的风和六月的雨，感受到美好的诗和远方。

后来，我在与试用的同学的讨论中发现这种通过图片播放音乐的功能还可以应用于启蒙教育等情景，让小朋友们能够调动多种感官来沉浸式地认识世界。于是，我在识别风景播放音乐的基础上，还增加了识别动物和识别天气的功能。同时，考虑到在一些场合不方便使用语音控制，我还增加了手势控制的功能，这样既拓展了项目的应用空间，同时也增加了使用者的交互体验。

## 项目设计

本项目可以通过深度学习识别到上传的照片中风景、动物或者天气的类别，再根据识别出的类别播放出相应的音乐。考虑到视障群体使用电脑的不便，我使用了音频识别技术，从而允许使用者通过语音来直接操控电脑，不必使用鼠标，并同时兼容了中文和英文两种语言。考虑到在一些公共场合使用语音的不便，以及应用于启蒙教育时婴幼儿语言能力不足等因素，我增加了手势识别技术，从而允许使用者通过几个常用简单的手势就能完成对有声相册的全部控制功能。

## 代码说明

本项目利用Flask框架实现了Web端部署，所以代码分为前端和后端两个部分。仓库提交了前端和后端的核心代码。

在前端代码部分，index.html为程序的主页面，scenery.html、animal.html和weather.html为程序三个模式对应的子级页面，box.css用于控制部分页面组件的样式。此外，还有部分用于页面响应的.js文件由于不是主干代码没有列出。

在后端代码部分，main.py是主程序文件，运行主程序即可启动程序的服务器。train.py是进行深度学习时使用的文件，Utils.py用于在上传图像后用于对图像进行预处理，addtext.py用于添加在图片右侧显示的文字，hand.py文件中是一些本项目使用到的基于OpenCV进行手势识别的函数。

项目结构：

```
Audio-Album
├─ back-end
│    ├─ Utils.py
│    ├─ addtext.py
│    ├─ hand.py
│    ├─ main.py
│    └─ train.py
└─ front-end
       ├─ animal.html
       ├─ box.css
       ├─ index.html
       ├─ scenery.html
       └─ weather.html
```

## 效果展示

![img\front-page.png](https://github.com/GoodMorningPeter/Audio-Album-Based-on-Speech-and-Gesture-Interaction/blob/main/img/front-page.png)

![img\animals.png](https://github.com/GoodMorningPeter/Audio-Album-Based-on-Speech-and-Gesture-Interaction/blob/main/img/animals.png)

![img\view.png](https://github.com/GoodMorningPeter/Audio-Album-Based-on-Speech-and-Gesture-Interaction/blob/main/img/view.png)

![img\gistures.png](https://github.com/GoodMorningPeter/Audio-Album-Based-on-Speech-and-Gesture-Interaction/blob/main/img/gistures.png)
