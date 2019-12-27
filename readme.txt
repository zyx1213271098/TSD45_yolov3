
	*此版本为原始版本*
	
	主要文件 train.py test.py detect_v1.py
	
	训练时：
	1.增加了只分1类的损失和label加载方式
	2.conf 的 Focal loss，但效果变差，主要是精度Precision变差很多
	3.坐标、长宽、分类、置信度损失权重保持原来的[8,4,1,64] 