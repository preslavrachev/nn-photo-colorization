# nn-photo-colorization
Can neural networks help bring a colourful life back to old black and white photos? Let's check it out!

_Note: This is a living document, summarising my research on the topic of photo colorization. It is far from its final version, but I would be more than happy, if it helps other people’s work. Feel free to send me comments, suggestions, or pull request changes_

Inspired, by [Emil Wallner's](https://twitter.com/EmilWallner) Medium post titled *["How to colorize black & white photos with just 100 lines of neural network code"](https://medium.freecodecamp.org/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d)*, I decided to delve into the field myself, and see what I might end up with.

# Challenges
## Choosing the right loss function
As it was reported by [1], when trained with a large set of images, the model becomes so generic that, in fact, colorised results come out mostly brownish, or very desaturated, as if a sepia filter has been applied on the photo. I have reported the same in my first versions of my experiment, while I  still used a simple generic CNN model. As it would turn out, this might have been caused by the choice of a loss function. I used a **mean squared error (mse)** loss. This is a good default loss function, when solving regression problems, where the input space is continuous. As research points out, however, the **ab** space is discrete. It can be confined to a number of 313 classes, which changes the nature of the problem from regression to classification. This requires a choice of a more appropriate loss function.

# References
[1] Wallner, E.: *How to colorize black & white photos with just 100 lines of neural network code* Medium.com, Oct 29, 2017 - https://medium.freecodecamp.org/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d

Cheng, Z., Yang, Q., Sheng, B.: *Deep colorization*. In: Proceedings of the IEEE
International Conference on Computer Vision. (2015) 415–423

Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." European Conference on Computer Vision. Springer, Cham, 2016.