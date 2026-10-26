# 同状态动作收益：首轮采集结果

共 129 个不同布局，2322 个 action×seed 标签；合法状态优化 1098 条，可行性恢复 1224 条。

每个 action 使用相同评价预算；初始评价计入预算，梯度、策略训练、公共合法化和额外验证分别计账。这是等评价次数实验，不能据此宣称等时间性能最优。RSS 是共享进程瞬时值，不是每臂峰值。

每个 Case 包含三个算法产生的 0/中点/终点状态并去重，同时另采合法化成功的 companion。同一 Case 的全部状态和几何版本应进入同一数据划分，禁止随机逐行切分造成泄漏。

`raw_delta_metrics` 是纯算法收益；`legalization_delta_metrics` 是后处理增量；`delta_metrics` 是完整 action+legalizer 收益。三者不得混用。非法输入的线长/热变化只作诊断，不会混入合法到合法的优化均值。恢复阶段 reward=合法性恢复指示+重叠下降；优化阶段 reward=归一化 HPWL 改善+0.3×归一化热代理改善。

热标签当前为未校准空间代理；HotSpot 适配和真实校准状态见 [热评估说明](../../hotspot_calibration.md)。

|Case/版本|阶段|配置|样本|原始合法率|后处理合法率|合法优化 ΔHPWL/mm|合法优化 Δ热代理|算法时间/s|合法化/s|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|Case1/original|feasibility_recovery|analytical-large|21|0|1|—|—|0.025056|0.00067326|
|Case1/original|feasibility_recovery|analytical-small|21|0.28571|1|—|—|0.024869|0.00063803|
|Case1/original|feasibility_recovery|rl-explore|21|0|1|—|—|0.011496|0.00077663|
|Case1/original|feasibility_recovery|rl-local|21|0|1|—|—|0.011355|0.00067556|
|Case1/original|feasibility_recovery|sa-explore|21|0.28571|1|—|—|0.008716|0.00052771|
|Case1/original|feasibility_recovery|sa-local|21|0.047619|1|—|—|0.0088802|0.00065629|
|Case1/original|legal_optimization|analytical-large|15|0.2|1|-2414|-0.00039838|0.024775|0.00055076|
|Case1/original|legal_optimization|analytical-small|15|0.4|1|-3913.6|0.003213|0.024828|0.00048572|
|Case1/original|legal_optimization|rl-explore|15|0.86667|1|-3314.5|-0.00072411|0.01114|0.00012887|
|Case1/original|legal_optimization|rl-local|15|1|1|-1765.8|-0.00057477|0.010913|6.1553e-05|
|Case1/original|legal_optimization|sa-explore|15|0.53333|1|-1635.4|1.683e-05|0.0085795|0.0003738|
|Case1/original|legal_optimization|sa-local|15|0.53333|1|-1842.2|0.0010269|0.0085072|0.00036385|
|Case10/original|feasibility_recovery|analytical-large|21|0.14286|1|—|—|0.064528|0.023734|
|Case10/original|feasibility_recovery|analytical-small|21|0|1|—|—|0.064007|0.023888|
|Case10/original|feasibility_recovery|rl-explore|21|0|1|—|—|0.079591|0.019857|
|Case10/original|feasibility_recovery|rl-local|21|0|1|—|—|0.079899|0.018606|
|Case10/original|feasibility_recovery|sa-explore|21|0|1|—|—|0.024687|0.019428|
|Case10/original|feasibility_recovery|sa-local|21|0|1|—|—|0.025158|0.019023|
|Case10/original|legal_optimization|analytical-large|21|0.14286|1|-505.99|-1.7708|0.078402|0.024831|
|Case10/original|legal_optimization|analytical-small|21|0|1|-1736.2|-0.89977|0.065271|0.02939|
|Case10/original|legal_optimization|rl-explore|21|0.7619|1|-474.83|-0.21362|0.084595|0.00758|
|Case10/original|legal_optimization|rl-local|21|0.52381|1|-374.96|-0.085736|0.087283|0.0087374|
|Case10/original|legal_optimization|sa-explore|21|0|1|749.18|-0.11204|0.032802|0.024255|
|Case10/original|legal_optimization|sa-local|21|0|1|178.96|0.058215|0.02618|0.019569|
|Case2/original|feasibility_recovery|analytical-large|21|0|0|—|—|0.026638|0.0011854|
|Case2/original|feasibility_recovery|analytical-small|21|0|0|—|—|0.026996|0.0012176|
|Case2/original|feasibility_recovery|rl-explore|21|0|0|—|—|0.011849|0.0012313|
|Case2/original|feasibility_recovery|rl-local|21|0|0|—|—|0.011844|0.0011022|
|Case2/original|feasibility_recovery|sa-explore|21|0|0|—|—|0.009298|0.001162|
|Case2/original|feasibility_recovery|sa-local|21|0|0|—|—|0.0092495|0.0011562|
|Case3/original|feasibility_recovery|analytical-large|21|0|1|—|—|0.05599|0.0010224|
|Case3/original|feasibility_recovery|analytical-small|21|0|1|—|—|0.055298|0.00098986|
|Case3/original|feasibility_recovery|rl-explore|21|0.33333|1|—|—|0.019618|0.00066097|
|Case3/original|feasibility_recovery|rl-local|21|0.14286|1|—|—|0.019774|0.00082765|
|Case3/original|feasibility_recovery|sa-explore|21|0.2381|1|—|—|0.016632|0.00076975|
|Case3/original|feasibility_recovery|sa-local|21|0.2381|1|—|—|0.016793|0.00074692|
|Case3/original|legal_optimization|analytical-large|21|0|1|-48051|0.11067|0.054804|0.0010028|
|Case3/original|legal_optimization|analytical-small|21|0.42857|1|-26463|0.083187|0.055196|0.00060323|
|Case3/original|legal_optimization|rl-explore|21|0.61905|1|-19658|0.022635|0.019414|0.00040964|
|Case3/original|legal_optimization|rl-local|21|0.57143|1|-12636|0.017739|0.019377|0.00044469|
|Case3/original|legal_optimization|sa-explore|21|0.61905|1|-3827.4|-0.047396|0.016517|0.00040146|
|Case3/original|legal_optimization|sa-local|21|0.80952|1|-2926.7|-0.017318|0.016371|0.00023882|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|analytical-large|21|0.14286|1|—|—|0.046279|0.001075|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|analytical-small|21|0|1|—|—|0.047023|0.0011595|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|rl-explore|21|0|1|—|—|0.017477|0.0010813|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|rl-local|21|0|1|—|—|0.017975|0.0011821|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|sa-explore|21|0|1|—|—|0.014215|0.0011132|
|Case4/case4-maskplace-59mm-alternative-v1|feasibility_recovery|sa-local|21|0|1|—|—|0.0147|0.0011459|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|analytical-large|21|0.14286|1|-21748|0.00067996|0.047197|0.0011416|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|analytical-small|21|0.14286|1|-8551.4|0.0072362|0.047166|0.0011555|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|rl-explore|21|1|1|-5043.4|-0.0048179|0.017675|6.0433e-05|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|rl-local|21|1|1|-2738.9|-0.0020163|0.017371|6.1133e-05|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|sa-explore|21|0.095238|1|2902.1|-0.0040237|0.014387|0.0010984|
|Case4/case4-maskplace-59mm-alternative-v1|legal_optimization|sa-local|21|0.095238|1|-59.439|-0.0014805|0.014416|0.0010516|
|Case5/original|feasibility_recovery|analytical-large|21|0.14286|1|—|—|0.052438|0.0014728|
|Case5/original|feasibility_recovery|analytical-small|21|0.42857|1|—|—|0.050392|0.00096344|
|Case5/original|feasibility_recovery|rl-explore|21|0.14286|1|—|—|0.01931|0.0013626|
|Case5/original|feasibility_recovery|rl-local|21|0|1|—|—|0.020246|0.0016212|
|Case5/original|feasibility_recovery|sa-explore|21|0.19048|1|—|—|0.015803|0.0013185|
|Case5/original|feasibility_recovery|sa-local|21|0.095238|1|—|—|0.015269|0.0014066|
|Case5/original|legal_optimization|analytical-large|21|0.14286|1|-37073|-0.00017781|0.057634|0.001601|
|Case5/original|legal_optimization|analytical-small|21|0.42857|1|-16957|-0.036008|0.055232|0.0010392|
|Case5/original|legal_optimization|rl-explore|21|0.7619|1|-4509.4|-0.024839|0.0211|0.00043833|
|Case5/original|legal_optimization|rl-local|21|0.61905|1|-4982|-0.090614|0.021831|0.00066932|
|Case5/original|legal_optimization|sa-explore|21|0.57143|1|1879.2|-0.11042|0.016956|0.00082253|
|Case5/original|legal_optimization|sa-local|21|0.7619|1|-1783.9|-0.087069|0.017009|0.0004813|
|Case6/original|feasibility_recovery|analytical-large|21|0.42857|1|—|—|0.047645|0.0020221|
|Case6/original|feasibility_recovery|analytical-small|21|0|1|—|—|0.044754|0.0030266|
|Case6/original|feasibility_recovery|rl-explore|21|0|1|—|—|0.020293|0.0029545|
|Case6/original|feasibility_recovery|rl-local|21|0|1|—|—|0.020889|0.002772|
|Case6/original|feasibility_recovery|sa-explore|21|0|1|—|—|0.015149|0.0027112|
|Case6/original|feasibility_recovery|sa-local|21|0|1|—|—|0.015387|0.0028332|
|Case6/original|legal_optimization|analytical-large|21|0|1|-12396|-0.29804|0.09762|0.0086383|
|Case6/original|legal_optimization|analytical-small|21|0.14286|1|-7120|-0.25073|0.098486|0.0067469|
|Case6/original|legal_optimization|rl-explore|21|0.7619|1|-2316.9|-0.036685|0.046658|0.0013664|
|Case6/original|legal_optimization|rl-local|21|0.61905|1|-2145.3|-0.020798|0.047034|0.0027305|
|Case6/original|legal_optimization|sa-explore|21|0.047619|1|3083.1|-0.097284|0.034288|0.0072296|
|Case6/original|legal_optimization|sa-local|21|0.28571|1|390.57|-0.075972|0.030536|0.0043719|
|Case7/original|feasibility_recovery|analytical-large|21|0.14286|1|—|—|0.08607|0.013452|
|Case7/original|feasibility_recovery|analytical-small|21|0.14286|1|—|—|0.0884|0.012591|
|Case7/original|feasibility_recovery|rl-explore|21|0|1|—|—|0.060923|0.01368|
|Case7/original|feasibility_recovery|rl-local|21|0|1|—|—|0.062305|0.014414|
|Case7/original|feasibility_recovery|sa-explore|21|0|1|—|—|0.035085|0.014653|
|Case7/original|feasibility_recovery|sa-local|21|0|1|—|—|0.041976|0.014771|
|Case7/original|legal_optimization|analytical-large|21|0.28571|1|-2064.6|-0.79614|0.097597|0.015213|
|Case7/original|legal_optimization|analytical-small|21|0.28571|1|-1510.7|-0.45973|0.10492|0.01468|
|Case7/original|legal_optimization|rl-explore|21|1|1|-531.98|-0.20912|0.058815|0.0002216|
|Case7/original|legal_optimization|rl-local|21|0.80952|1|-456.33|-0.1685|0.064117|0.0037273|
|Case7/original|legal_optimization|sa-explore|21|0.19048|1|503.91|-0.17881|0.036781|0.011773|
|Case7/original|legal_optimization|sa-local|21|0.28571|1|48.214|-0.068597|0.040926|0.011141|
|Case8/original|feasibility_recovery|analytical-large|18|0.16667|1|—|—|0.10671|0.025607|
|Case8/original|feasibility_recovery|analytical-small|18|0|1|—|—|0.11595|0.02923|
|Case8/original|feasibility_recovery|rl-explore|18|0|1|—|—|0.072255|0.024458|
|Case8/original|feasibility_recovery|rl-local|18|0|1|—|—|0.075435|0.025374|
|Case8/original|feasibility_recovery|sa-explore|18|0|1|—|—|0.044373|0.027565|
|Case8/original|feasibility_recovery|sa-local|18|0|1|—|—|0.04269|0.025467|
|Case8/original|legal_optimization|analytical-large|21|0.14286|1|-2552.7|-0.40336|0.055272|0.011457|
|Case8/original|legal_optimization|analytical-small|21|0.42857|1|-1477.6|-0.28577|0.051131|0.0084858|
|Case8/original|legal_optimization|rl-explore|21|0.80952|1|-318.93|-0.0059448|0.039783|0.0018863|
|Case8/original|legal_optimization|rl-local|21|0.71429|1|-188.5|-0.012897|0.034126|0.0026916|
|Case8/original|legal_optimization|sa-explore|21|0.047619|1|258.46|-0.016398|0.022391|0.01283|
|Case8/original|legal_optimization|sa-local|21|0.095238|1|9.5942|0.0052919|0.02156|0.010597|
|Case9/original|feasibility_recovery|analytical-large|18|0|1|—|—|0.066319|0.013849|
|Case9/original|feasibility_recovery|analytical-small|18|0|1|—|—|0.066775|0.011553|
|Case9/original|feasibility_recovery|rl-explore|18|0|1|—|—|0.052915|0.0086466|
|Case9/original|feasibility_recovery|rl-local|18|0|1|—|—|0.053924|0.0085776|
|Case9/original|feasibility_recovery|sa-explore|18|0|1|—|—|0.023091|0.0086323|
|Case9/original|feasibility_recovery|sa-local|18|0|1|—|—|0.022328|0.008702|
|Case9/original|legal_optimization|analytical-large|21|0.28571|1|-4839.6|-0.91218|0.065273|0.0095852|
|Case9/original|legal_optimization|analytical-small|21|0.28571|1|-5183.4|-0.45471|0.064542|0.0093387|
|Case9/original|legal_optimization|rl-explore|21|0.85714|1|-1766.9|-0.079698|0.051995|0.0011479|
|Case9/original|legal_optimization|rl-local|21|0.80952|1|-1000.7|-0.063312|0.052662|0.0017289|
|Case9/original|legal_optimization|sa-explore|21|0.095238|1|2582.6|-0.038507|0.02315|0.008913|
|Case9/original|legal_optimization|sa-local|21|0.047619|1|784.98|0.027683|0.022762|0.0093479|

原始逐动作 JSONL、状态坐标、源输入哈希和各阶段调用数在各 Case 目录；汇总见 [summary.csv](summary.csv)、[summary.json](summary.json)，预算核验见 [budget_audit.json](budget_audit.json)。

当前是有限配置全臂数据采集，尚未训练 CNN/Bandit。三个本地算法不等同于论文官方实现；配置差异是机制可识别性测试，不预设某算法适合某阶段，也不把公共合法化收益归功于某算法。
