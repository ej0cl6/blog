--- 
published: true
title: GAN and Conditional GAN
layout: post
category: Machine Leaning
---

一直以來我都想寫一些類似科普的文章，希望能夠用沒那麼技術的文字來介紹一些東西  
只是之前都太懶惰了哈哈哈哈
最近決定邊讀新的paper，邊用blog介紹這些paper，寫下我對這些paper的理解  
有發現錯誤的地方請跟我說，也歡迎大家討論  

這篇要介紹的是machine learning現在超紅的generative adversarial nets (GAN) [1] 以及他的延伸   conditional generative adversarial nets (Conditional GAN) [2]  
適合對machine learning有些認識的人閱讀  

<!-- more -->

在machine learning的世界裡，我們通常會準備一些data  
然後設計machine learning algorithm來讓machine從這些data中產生一個model學東西  
比如說，我們準備了很多張小狗的圖片，希望machine能夠學一個"辨識小狗"的model  
這個model對於新的圖片，要能夠判斷這是不是一張小狗的圖片  
這個model可以回答yes or no，也可以回答這是小狗圖片的機率是多少  
像這樣的model，我們稱之為discriminative model  
透過這個model，machine能認得小狗的樣子，但如果我們叫machine畫一隻新的小狗，machine可能畫不出來  
換句話說，machine可以分辨(discriminate)，可是無法產生(generate)  

因此有一部份的人在研究generative model，希望model能夠學會產生data  
用剛剛的例子，machine需要從一堆小狗的圖片當中學一個model，這個model要能畫出新的小狗  
要學一個generative model通常會比discriminative model困難一些  

這篇的主角，generative adversarial nets (GAN) 用了一個很有趣的方法來讓machine學generative model  
GAN設計了兩個neural network: generator和discriminator，並且讓他們互相競爭來達到目的  
整個GAN的架構如下

![gan](/blog/assets/img/gan.png)

假設我們現在有一些data $$\{x_1, x_2, ..., x_n\}$$  
我們的目標是學一個generator $$G$$，這個 $$G$$ 要能將一個任意的random noise $$z$$ 變成新的data  
換句話說，對於一個任意的random noise $$z$$， $$G(z)$$ 就是所產生的data  

這時有另一個discriminator $$D$$，他的目標是要分辯哪個是真的data，哪個是 $$G$$ 產生的data  
也就是說，對於一個data $$x$$，$$x$$ 有可能是真的data，也有可能是 $$G$$ 產生的  
而 $$D(x)$$ 則是discrminator覺得這個 $$x$$ 是真的data的機率是多少  
如果用數學式來表達，discriminator $$D$$ 要做的事情是  

$$\max\limits_{D} V(D,G) = \mathbb{E}_{x_i} \left[\log D(x) \right] + \mathbb{E}_{z}    \left[ \log (1-D(G(z))) \right] $$  

這邊解釋一下這個式子  
對於真的data $$x_i$$，$$D(x)$$應該要越高越好，所以 $$\mathbb{E}_{x_i} \left[\log D(x) \right]$$ 要越大  
對於 $$G$$ 產生的 $$G(z)$$，$$D(G(z))$$要越低越好，所以 $$\mathbb{E}_{z}  \left[ \log (1-D(G(z)))   \right]$$ 要越大

而generator $$G$$ 要做的事情則是反過來，要產生data讓discriminator $$D$$ 無法分辨  
也就是  

$$\min\limits_{G} \max\limits_{D} V(D,G) = \mathbb{E}_{x_i} \left[\log D(x) \right] +   \mathbb{E}_{z}  \left[ \log (1-D(G(z))) \right] $$  

整個流程就是generator $$G$$ 想辦法產生data，discriminator $$D$$ 想辦法分辨data  
$$G$$ 和 $$D$$ 不斷競爭，最後到底誰會贏呢  
在paper中，作者證明了最後 $$D$$ 會落敗，最後不管什麼 $$x$$，$$D(x)$$ 都會是0.5  
也就是我們的generator $$G$$ 能夠產生很逼真的data  

這邊放幾個paper的實驗結果  
第一個是給machine一堆人臉的圖片，要machine畫出新的人臉  
除了最右邊那欄是真的圖片，其他的都是machine畫出來的，還算逼真吧  

![gan_exp1](/blog/assets/img/gan_exp1.png)

這是要machine畫數字

![gan_exp2](/blog/assets/img/gan_exp2.png)

GAN透過競爭機制，成功得讓machine學會generative model  
不過這邊有個小問題，在第二個實驗當中，我們給machine的數字有很多種  
machine雖然能夠產生逼真的數字，可是我們卻無法指定machine產生特定的數字  
這也成了conditional generative adversarial nets (Conditional GAN) 這篇paper的動機  


Conditional GAN想要解決的問題是這樣的  
現在我們一樣給machine很多data，不過每個data會對應一個label  
以數字圖的例子來說，每個data就是一張圖，而對應的label就是這張圖代表的數字  
我們一樣希望machine能夠從這些圖中學一個generative model，並且能夠畫出新的數字圖  
不一樣的是，我們希望可以指定畫出來的數字要是什麼  
也就是，我們可以指定label要是什麼  


Conditional GAN的架構跟GAN非常類似  
差別只在於generator和discriminator的input多了label  

![conditional_gan](/blog/assets/img/conditional_gan.png)

在這個架構下，generator會根據不同的label $$y$$ 產生不同的data $$G(z|y)$$  
而discriminator $$D$$ 預測的機率 $$D(x|y)$$ 則是代表這個 $$x$$ 的label是 $$y$$ 的機率  
Conditional GAN的數學式跟GAN的數學式非常像  

$$\min\limits_{G} \max\limits_{D} V(D,G) = \mathbb{E}_{x_i} \left[\log D(x|y) \right]   + \mathbb{E}_{z}  \left[ \log (1-D(G(z|y))) \right] $$

透過這樣的改動，我們就可以指定machine產生data的label了  

下圖是要求machine畫各式各樣的0~9  

![conditional_gan_exp](/blog/assets/img/conditional_gan_exp.png)

GAN和Conditional GAN有很多的應用，等我之後看了更多的paper再來介紹給大家  


**Reference**  
1. I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, and B. Xu, D. Warde-Farley, S. Ozair, A.   Courville, and Y. Bengio. "Generative adversarial nets." NIPS, 2014.  
2. M. Mirza and S. Osindero. "Conditional generative adversarial nets." Preprint arXiv:1411.1784, 2014.  
