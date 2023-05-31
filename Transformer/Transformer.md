# Transformer

## object detection

### DETR

![image-20230531092919167](Transformer.assets/image-20230531092919167.png)

![image-20230531100639661](Transformer.assets/image-20230531100639661.png)

object queries做自注意力主要是为了移除冗余框，因为通过注意力机制互相通信后，就可以知道每个query会得到一个什么样的框，而互相之间不要去做重复的框。

细节：

在decoder的每层都

loss

最优二分图匹配

![image-20230531095449193](Transformer.assets/image-20230531095449193.png)

![image-20230531095959607](Transformer.assets/image-20230531095959607.png)

### ViLT

![image-20230531141633211](Transformer.assets/image-20230531141633211.png)

精确度可能还不够，但是速度快了上千倍

![image-20230531151417098](Transformer.assets/image-20230531151417098.png)

![image-20230531151513918](Transformer.assets/image-20230531151513918.png)

![image-20230531152916452](Transformer.assets/image-20230531152916452.png)

![image-20230531153520910](Transformer.assets/image-20230531153520910.png)