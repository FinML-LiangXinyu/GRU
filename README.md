# $GRU$ 算法原理

对于长度为 $T$ 的时间序列数据 $x=\left[x_1,x_2,\ldots,x_t,\ldots,x_T\right]$ ， $x_t$ 为时刻 $t$ 的输入向量。 $GRU$ 算法的结构单元如下：

$r_t=\sigma\left(U_rh_{t-1}+W_rx_t+b_r\right)$

$z_t=\sigma\left(U_zh_{t-1}+W_zx_t+b_z\right)$

$\widetilde{h_t}=tanh(U_h\left(r_t\odot h_{t-1}\right)+W_hx_t+b_h)$

$h_t=(1-z_t)\odot\widetilde{h_t}+z_t\odot h_{t-1}$

$c_t=Vh_t+b_c$

$\widehat{y_t}=softmax\left(c_t\right)$

其中， $h_{t-1}$ 代表 $t-1$ 时刻的隐状态， $x_t$ 为时刻 $t$ 的输入， $r_t$ 为时刻 $t$ 的重置门，它决定了 $h_{t-1}$ 中的哪些信息需要被重置， $z_t$ 为时刻 $t$ 的更新门，它决定了 $h_{t-1}$ 中哪些信息需要被 $\widetilde{h_t}$ 更新。时刻 $t$ 的 $\widetilde{h_t}$ 经过 $tanh(·)$ 激活后，在更新门 $zt$ 的控制下转换为 $t$ 时刻的隐状态 $h_t$ ，时刻 $t$ 的净输入 $c_t$ 经过 $softmax(·)$ 转换为时刻 $t$ 的最终输出 $\widehat{y_t}$ ， $U_r$ 、 $U_z$ 、 $U_h$ 、 $W_r$ 、 $W_z$ 、 $W_h$ 、 $V$ 为神经网络的权重矩阵， $b_r$ 、 $b_z$ 、 $b_h$ 、 $b_c$ 为神经络的偏置向量。

矩阵 $U_r$ 、 $U_z$ 、 $U_h$ 、 $W_r$ 、 $W_z$ 、 $W_h$ 在每个时刻 $t$ 都会影响该时刻的隐状态 $h_t$ 进而影响总损失 $L$ 。

$U_r$ 对应前向链式传播路径：

$U_r\rightarrow r_1,r_2,\ldots,r_t,\ldots,r_T\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow L$

可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot d\widetilde{h_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\odot U_h\left(dr_t\odot h_{t-1}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)^TU_h\left(dr_t\odot h_{t-1}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)^Tdr_t\odot h_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\right)^Tdr_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\right)^T{\sigma_{r_t}}^\prime\odot d U_rh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\left(\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){h_{t-1}}^T\right)^TdU_r}\right)$

$\frac{\partial L}{\partial U_r}=\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){h_{t-1}}^T}$

同理：

$\frac{\partial L}{\partial W_r}=\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){x_t}^T}$

$\frac{\partial L}{\partial b_r}=\sum_{t=1}^{T}{\left({U_h}^T\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime}$

$U_z$ 对应前向链式传播路径：

$U_z\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow L$

可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdz_t\odot\left(h_{t-1}-\widetilde{h_t}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T{\sigma_{z_t}}^\prime\odot d U_zh_{t-1}\odot\left(h_{t-1}-\widetilde{h_t}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\left(\frac{\partial L}{\partial h_t}\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)\right){h_{t-1}}^T\right)^TdU_z}\right)$

$\frac{\partial L}{\partial U_z}=\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)\right){h_{t-1}}^T}$

同理：

$\frac{\partial L}{\partial W_z}=\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)\right){x_t}^T}$

$\frac{\partial L}{\partial b_z}=\sum_{t=1}^{T}{\frac{\partial L}{\partial h_t}\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)}$

$U_h$ 对应前向链式传播路径：

$U_h\rightarrow\widetilde{h_1},\widetilde{h_2},\ldots,\widetilde{h_t},\ldots,\widetilde{h_T}\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow L$

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot d\widetilde{h_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\odot d U_h\left(r_t\odot h_{t-1}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)^TdU_h\left(r_t\odot h_{t-1}\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\left(r_t\odot h_{t-1}\right)^T\right)^TdU_h}\right)$

$\frac{\partial L}{\partial U_h}=\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\left(r_t\odot h_{t-1}\right)^T}$

$W_h$ 对应前向链式传播路径：

$W_h\rightarrow\widetilde{h_1},\widetilde{h_2},\ldots,\widetilde{h_t},\ldots,\widetilde{h_T}\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow L$

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot d\widetilde{h_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\odot d W_hx_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right){x_t}^T\right)^TdW_h}\right)$

$\frac{\partial L}{\partial W_h}=\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right){x_t}^T}$

同理：

$\frac{\partial L}{\partial b_h}=\sum_{t=1}^{T}{\frac{\partial L}{\partial h_t}\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)}$

$t$ 时刻的隐状态 $h_t$ 会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，并通过影响 $t+1$ 时刻的隐状态 $h_{t+1}$ 影响以后时刻的损失 $L_{>t}$ 进而影响总损失 $L$ 。对应链式传播路径为：

$h_t\rightarrow c_t\rightarrow L_t\rightarrow L$

$h_t\rightarrow h_{t+1}\rightarrow L$

$dh_{t+1}=(1-z_{t+1})\odot d\widetilde{h_{t+1}}+z_{t+1}\odot dh_t+dz_{t+1}\odot(h_t-\widetilde{h_{t+1}})=(1-z_{t+1})\odot\left(1-\widetilde{h_{t+1}}\odot\widetilde{h_{t+1}}\right)\odot U_hr_{t+1}\odot dh_t+z_{t+1}\odot dh_t+(h_t-\widetilde{h_{t+1}})\odot{\sigma_{z_{t+1}}}^\prime\odot U_zdh_t$

$dL=tr\left(\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t\right)+tr\left(\left(\frac{\partial L}{\partial h_{t+1}}\right)^Tdh_{t+1}\right)=tr\left(\left(V^T\left(\widehat{y_t}-y_t\right)\right)^Tdh_t\right)+tr\left(\left(\frac{\partial L}{\partial h_{t+1}}\right)^T\left(\left(1-z_{t+1}\right)\odot\left(1-\widetilde{h_{t+1}}\odot\widetilde{h_{t+1}}\right)\odot U_hr_{t+1}\odot d h_t+z_{t+1}\odot d h_t+\left(h_t-\widetilde{h_{t+1}}\right)\odot{\sigma_{z_{t+1}}}^\prime\odot U_zdh_t\right)\right)=tr\left(\left(V^T\left(\widehat{y_t}-y_t\right)\right)^Tdh_t\right)+tr\left(\left(\frac{\partial L}{\partial h_{t+1}}\odot\left(1-z_{t+1}\right)\odot\left(1-\widetilde{h_{t+1}}\odot\widetilde{h_{t+1}}\right)\odot U_hr_{t+1}\right)^Tdh_t\right)+tr\left(\left(\frac{\partial L}{\partial h_{t+1}}\odot z_{t+1}\right)^Tdh_t\right)+tr\left(\left({U_z}^T\frac{\partial L}{\partial h_{t+1}}\odot\left(h_t-\widetilde{h_{t+1}}\right)\odot{\sigma_{z_{t+1}}}^\prime\right)^Tdh_t\right)$

$\frac{\partial L}{\partial h_t}=V^T\left(\widehat{y_t}-y_t\right)+\frac{\partial L}{\partial h_{t+1}}\odot\left(1-z_{t+1}\right)\odot\left(1-\widetilde{h_{t+1}}\odot\widetilde{h_{t+1}}\right)\odot U_hr_{t+1}+\frac{\partial L}{\partial h_{t+1}}\odot z_{t+1}+{U_z}^T\frac{\partial L}{\partial h_{t+1}}\odot\left(h_t-\widetilde{h_{t+1}}\right)\odot{\sigma_{z_{t+1}}}^\prime$

此外，有：

$\frac{\partial L}{\partial h_T}=V^T\left(\widehat{y_T}-y_T\right)$

记 $\delta_t=\frac{\partial L}{\partial h_t}$ ，则有：

$\delta_T=V^T\left(\widehat{y_T}-y_T\right)$

$\delta_t=V^T\left(\widehat{y_t}-y_t\right)+\delta_{t+1}\odot\left(1-z_{t+1}\right)\odot\left(1-\widetilde{h_{t+1}}\odot\widetilde{h_{t+1}}\right)\odot U_hr_{t+1}+\delta_{t+1}\odot z_{t+1}+{U_z}^T\delta_{t+1}\odot\left(h_t-\widetilde{h_{t+1}}\right)\odot{\sigma_{z_{t+1}}}^\prime$

反向传播公式汇总如下：

$\frac{\partial L}{\partial U_r}=\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){h_{t-1}}^T}$

$\frac{\partial L}{\partial W_r}=\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){x_t}^T}$

$\frac{\partial L}{\partial W_r}=\sum_{t=1}^{T}{\left(\left({U_h}^T\left(\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\right)\odot h_{t-1}\odot{\sigma_{r_t}}^\prime\right){x_t}^T}$

$\frac{\partial L}{\partial U_z}=\sum_{t=1}^{T}{\left(\delta_t\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)\right){h_{t-1}}^T}$

$\frac{\partial L}{\partial W_z}=\sum_{t=1}^{T}{\left(\delta_t\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)\right){x_t}^T}$

$\frac{\partial L}{\partial b_z}=\sum_{t=1}^{T}{\delta_t\odot{\sigma_{z_t}}^\prime\odot\left(h_{t-1}-\widetilde{h_t}\right)}$

$\frac{\partial L}{\partial U_h}=\sum_{t=1}^{T}{\left(\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right)\left(r_t\odot h_{t-1}\right)^T}$

$\frac{\partial L}{\partial W_h}=\sum_{t=1}^{T}{\left(\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)\right){x_t}^T}$

$\frac{\partial L}{\partial b_h}=\sum_{t=1}^{T}{\delta_t\odot\left(1-z_t\right)\odot\left(1-\widetilde{h_t}\odot\widetilde{h_t}\right)}$

$\frac{\partial L}{\partial V}=\sum_{t=1}^{T}{\left(\widehat{y_t}-y_t\right){h_t}^T}$

$\frac{\partial L}{\partial b_c}=\sum_{t=1}^{T}\left(\widehat{y_t}-y_t\right)$
