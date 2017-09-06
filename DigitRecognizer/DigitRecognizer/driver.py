import model1
import model2

model_save_path1 = "model/model.ckpt" 
model_save_path2 = "model/model2.ckpt" 
model_save_path3 = "model/model3.ckpt" 
model_save_path4 = "model/model4.ckpt" 
model_save_path5 = "model/model5.ckpt" 

#Three of the first model
model1.TrainConvNet(model_save_path1)
model1.TrainConvNet(model_save_path2)
model1.TrainConvNet(model_save_path3)

#Four of the second model
model2.TrainConvNet(model_save_path4);
model2.TrainConvNet(model_save_path5);
model2.TrainConvNet(model_save_path6);
model2.TrainConvNet(model_save_path7);