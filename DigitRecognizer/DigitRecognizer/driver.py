import model1
import model2
import numpy as np

model_save_path1 = "model/model.ckpt" 
model_save_path2 = "model/model2.ckpt" 
model_save_path3 = "model/model3.ckpt" 
model_save_path4 = "model/model4.ckpt" 
model_save_path5 = "model/model5.ckpt" 
model_save_path6 = "model/model6.ckpt" 
model_save_path7 = "model/model7.ckpt" 

##Three of the first model
#model1.TrainConvNet(model_save_path1)
#model1.TrainConvNet(model_save_path2)
#model1.TrainConvNet(model_save_path3)

##Four of the second model
#model2.TrainConvNet(model_save_path4);
#model2.TrainConvNet(model_save_path5);
#model2.TrainConvNet(model_save_path6);
#model2.TrainConvNet(model_save_path7);


results1 = model1.LoadAndRun(model_save_path1)
results2 = model1.LoadAndRun(model_save_path2)
results3 = model1.LoadAndRun(model_save_path3)
results4 = model1.LoadAndRun(model_save_path4)

results5 = model2.LoadAndRun(model_save_path5)
results6 = model2.LoadAndRun(model_save_path6)
results7 = model2.LoadAndRun(model_save_path7)


with open("results/results.csv", 'w') as file:
    file.write("ImageId,Label\n")
    for idx in range(len(results1)):
         pred1 = results1[idx]
         pred2 = results2[idx]
         pred3 = results3[idx]
         pred4 = results4[idx]
         pred5 = results5[idx]
         pred6 = results6[idx]
         pred7 = results7[idx]

         nums = np.zeros(10)
        
         nums[pred1] = nums[pred1] + 1 
         nums[pred2] = nums[pred2] + 1 
         nums[pred3] = nums[pred3] + 1 
         nums[pred4] = nums[pred4] + 1 
         nums[pred5] = nums[pred5] + 1 
         nums[pred6] = nums[pred5] + 1 
         nums[pred7] = nums[pred5] + 1 

         prediction = np.argmax(nums)

         file.write(str(idx + 1))
         file.write(",")
         file.write(str(prediction))
         file.write("\n")