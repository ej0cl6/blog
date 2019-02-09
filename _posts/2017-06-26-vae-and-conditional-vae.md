--- 
published: true
title: VAE and Conditional VAE
layout: post
category: Machine Leaning
---

今天要來介紹另一種generative model──variational auto-encoder (VAE) [1] 以及他的延伸 conditional variational auto-encoder (Conditional VAE) [2]  
之前提到的generative adversarial nets (GAN)   是讓discriminator和generator透過互相競爭產生一個足夠好的generator來generate data  
而VAE則是想透過機率模型模擬出data的distribution  
新的data可以透過distribution sample得到  

<!-- more -->

我們先來看看傳統的auto-encoder吧

![auto_encoder](/blog/assets/img/auto_encoder.png)

傳統的auto-encoder有兩個東西要學──encoder和decoder  
通常我們用neural network來學encoder和decoder，並希望encoder和decoder滿足一些性質  

data $$x$$ 可以透過encoder轉成一個latent vector $$z$$  
這個latent vector $$z$$ 可以想像成是data $$x$$ 的另一種表示方法  
舉個動物圖片的例子來說，我們可以將各種動物的圖片透過encoder轉成latent vector $$z$$  
$$z$$ 每個維度可以代表不同的意義，像是動物有幾隻腳，有沒有翅膀等等，當然可能會有更複雜的意義，要看encoder學成怎麼樣  
用latent vector $$z$$ 來代替data $$x$$ 有一些好處，其中一個是我們可以讓latent vector $$z$$ 的維度比data $$x$$ 小很多，這樣用 $$z$$ 代替 $$x$$ 在很多運算上都會比較快速  

而latent vector $$z$$ 可以透過decoder轉回data $$\bar{x}$$  
為了確保latent vector $$z$$ 能夠完整的保存原本的data $$x$$，我們會希望data $$x$$ 透過encoder轉成latent vector $$z$$ 再透過decoder轉回的 $$\bar{x}$$ 能跟原本的 $$x$$ 越像越好  
這大致上就是傳統的auto-encoder  

那麼auto-encoder有沒有辦法產生任意的data呢  
一個簡單的想法是，我們隨意指定一個latent vector $$z$$，透過decoder將 $$z$$ 轉成 $$\bar x$$ 就可產生一個新的data了  
不過不幸地，這個想法產生的data效果並不好  
舉剛剛動物圖片的例子，常常隨意指定的 $$z$$ 所產生的圖片看起來並不像一個動物  
最主要的原因是這樣學出來 $$z$$ 的space其實非常的sparse，我們隨意指定的 $$z$$ 不一定很有意義  

variational auto-encoder (VAE) 剛好可以解決這個問題  
我們現在換一種機率的觀點來看 $$x$$ 和 $$z$$ 之間的關係  
假設真實世界的的data $$x$$ 是根據latent vector $$z$$ 產生的，而 $$z$$ 則是根據某種distribution $$P(z)$$ 產生的  
換句話說，我們根據 $$P(z)$$ sample出latent vector $$z$$，再根據 $$P(x|z)$$ 得到data $$x$$  
為了簡單起見，VAE的paper假設 $$P(z)=\mathcal N (z; 0, I)$$ 是一個multivariate normal distribution  

這時候我們回來看看剛剛auto-encoder的架構

![prob_auto_encoder](/blog/assets/img/prob_auto_encoder.png)

有了data $$x$$，可以透過 $$P(z|x)$$ 得到 latent vector $$z$$  
如果我們能用neural network學一個distribution $$Q(z|x) \approx P(z|x)$$，那麼這個network就可以看成是encoder  
VAE的paper假設我們要學的encoder $$Q(z|x) = \mathcal N (z; \mu, \sigma^2 I)$$ 是一個multivariate Gaussian  
換句話說，對於data $$x$$，encoder $$Q(z|x)$$ 會output一個 $$z$$ 的multivariate Gaussian distribution  
有了這個distribution，我們就可以sample一個可能的的latent vector $$z$$ 並配合neural network學出 $$P(x|z)$$ 當做decoder  
整個架構會像這樣  

![vae](/blog/assets/img/vae.png)

在VAE的paper中，是用一個叫做variational inference的方法來學encoder $$Q(z|x)$$  
由於variational inference的數學式子比較複雜，以後有空再介紹，有興趣的人可以看這裡  

一旦我們有了encoder和decoder，我們便可以從 $$P(z)=\mathcal N (z; 0, I)$$ sample出 $$z$$，再透過decoder產生新的data  
這樣的機率模型解決了latent space太sparse的問題  
我們來看看VAE根據不同的 $$z$$ 產生出來的data長怎麼樣  

![vae_exp](/blog/assets/img/vae_exp.png)

這是VAE從一堆數字圖中所學出來的generative model，的確可以產生出新的數字圖  
不過和GAN一樣，VAE並沒有辦法指定data的label  
因此有人提出了VAE的延伸──conditional variational auto-encoder (Conditional VAE)  
延伸的方法和Conditional GAN一樣，將encoder和decoder的input多給一個label的資訊  
架構如下  

![conditional_vae](/blog/assets/img/conditional_vae.png)

如此一來我們就可以指定產生data的label

![conditional_vae_exp1](/blog/assets/img/conditional_vae_exp1.png)

在Conditional VAE的paper中還有一個有趣的實驗  

![conditional_vae_exp2](/blog/assets/img/conditional_vae_exp2.png)


上圖中，最左一欄的數字是真的data  
將這些真的data透過encoder可以得到對應的latent vector $$z$$  
右邊的0~9則是這些latent vector在decode時input設成不同label的結果  
結果非常有趣，不管latent vector是從哪個數字得到的，decode出來的數字都會是decoder的input   label，差別只在於字形的不同  

以上大致上就是VAE以及Conditional VAE的介紹

**Reference**

1. D. P. Kingma and M.Welling. "Auto-encoding variational bayes." Preprint arXiv:1312.6114, 2013.
2. D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling. "Semi-supervised learning with deep generative models." NIPS, 2014.
3. http://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
4. http://zhuanlan.zhihu.com/p/22464760
5. http://www.cnblogs.com/wangxiaocvpr/p/6231019.html
6. http://kvfrans.com/variational-autoencoders-explained/



