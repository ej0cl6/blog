--- 
published: true
title: 主人與他的11個客人 
layout: post
category: Math
---

好久沒更新blog了，今天要來和大家分享一個數學面試題  
這個面試題是我聽過的題目中數一數二有趣的題目  
  
題目如下：  
有一個主人邀請了11個客人來他家玩  
在客人回去之前，主人決定送其中一個客人一份禮物  
主人想了一個遊戲來決定這個幸運的人，他要所有客人和自己圍成一圈  
一開始禮物在主人手上，接著他會有一半的機率把禮物往左傳，一半的機率往右傳  
拿到禮物的人也要以一半往左一半往右的機率把禮物傳出去  
只要客人一碰到這個禮物，他就喪失拿到禮物的機會了，不過他還是得繼續幫忙傳禮物  
換句話說，比其他10個客人晚碰到禮物的那個客人即可獲得禮物  
現在問題來了，如果你是其中一個客人，請問你坐在圓圈中哪個位子獲得禮物的機會最高呢?  

<!-- more -->

<input onclick="ans.style.display=ans.style.display=='none'?'':'none'" type="checkbox" value="ON" />點我看參考解答

<div id="ans" style="display: none;">
應該大多數人的第一感都是主人對面的那個位子吧XD<br>  
我們來看看會發生什麼事情<br>
<br>
如下圖，假設將圓圈座位編號0到11，而且主人做在0號的位子<br>

<img src="{{ site.baseurl }}/assets/img/guest_circle.png" height="300"><br>

禮物會從主人手中開始傳，也就是0號位子<br>
<br>
我們來先來算算看1號客人可以拿到禮物的機率是多少<br>
1號客人要能拿到禮物的條件是，禮物從0號位子傳出去後，經過來來回回傳遞，最後2號客人比1號客人早碰到禮物<br>
換句話說，就是從1號客人的右手邊開始傳，但是左手邊的人要比自己還先碰到禮物的機率<br>
假設這個機率叫做P1吧<br>
<br>
在仔細計算P1等於多少前，我們來看看其他客人拿到禮物的機率跟P1有甚麼關係<br>
因為對稱性的關係，11號客人拿到禮物的機率顯然也是P1<br>
<br>
那麼4號客人拿到禮物的機率會怎麼樣呢<br>
在4號客人碰到禮物前，會有兩種情況發生：<br>
(a) 3號客人比5號客人早碰到禮物<br>
(b) 5號客人比3號客人早碰到禮物<br>
假設這兩個情況的機率為Pa和Pb，那麼Pa+Pb = 1<br>
<br>
如果遇到了(a)的狀況，這時候禮物會從3號客人，也就是4號客人的右手邊傳出去<br>
而4號客人要拿到禮物的條件是比他左手邊的人還要晚碰到禮物<br>
這個情況有沒有很熟悉呢?<br>
沒錯! 這跟剛剛計算1號客人的情況很像，在(a)的狀況下，4號客人拿到禮物的機率就是P1<br>
所以透過情況(a)，4號客人拿到禮物的機率為Pa*P1<br>
<br>
那情況(b)會怎麼樣呢<br>
當情況(b)發生時，這時候禮物會從5號客人，也就是4號客人的左手邊傳出去<br>
而4號客人要拿到禮物的條件是比他右手邊的人還要晚碰到禮物，這個11號客人遇到的情況也很像，也是P1<br>
所以透過情況(b)，4號客人拿到禮物的機率為Pb*P1<br>
<br>
綜合以上的兩個情況，4號客人拿到禮物的機率為Pa*P1 + Pb*P1 = (Pa + Pb)*P1 = P1<br>
這是一個令人稍為驚訝的結果，明明1號客人離主人位子比較接近，但是4號客人拿到禮物的機率卻跟1號客人一樣<br>
<br>
如果我們用同樣的方式去計算其他客人拿到禮物的機率，會發現所有客人的機率都是P1<br>
所以這題的結論是，坐哪個位子都沒有差別，拿到的禮物機率都是1/11<br>
<br>
這真是我遇過的問題中數一數二有趣的題目了哈哈哈哈<br>
</div>
