[TOC]

# Reinforcement Learning

## Algorithms

| Algorithm           | Game                     | Library | Code |
| :------------------ | ------------------------ | ------- | :--: |
| REINFORCE Method    | CartPole-v0              | Keras   |  Y   |
| N-Step A2C          | BreakoutDeterministic-v4 | Pytorch |  Y   |
| N-Step A3C with GAE | SuperMarioBros-v0        | Pytorch |  Y   |

## Examples

### Super Mario N-Step A3C with GAE

![](images/super-mario.gif)








# How to convert video to GIF file

``````
mkdir frames
ffmpeg -i flappybird.mp4 -qscale:v 2  -r 25 'frames/frame-%05d.jpg'
cd frames
convert -delay 4 -loop 0 *.jpg flappybird.gif
```

FFMpeg and Imagemagic(Convert command) have the following options.

```
-r 5 stands for FPS value
    for better quality choose bigger number
    adjust the value with the -delay in 2nd step
    to keep the same animation speed

-delay 20 means the time between each frame is 0.2 seconds
   which match 5 fps above.
   When choosing this value
       1 = 100 fps
       2 = 50 fps
       4 = 25 fps
       5 = 20 fps
       10 = 10 fps
       20 = 5 fps
       25 = 4 fps
       50 = 2 fps
       100 = 1 fps
       in general 100/delay = fps

-qscale:v n means a video quality level where n is a number from 1-31, 
   with 1 being highest quality/largest filesize and 
   31 being the lowest quality/smallest filesize.

-loop 0 means repeat forever
```