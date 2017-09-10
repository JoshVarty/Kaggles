import model1
import model2
import model3
import numpy as np

model_save_path1 = "model/model.ckpt" 
model_save_path2 = "model/model2.ckpt" 
model_save_path3 = "model/model3.ckpt" 

###Three of the first model
model1.TrainConvNet(model_save_path1)
model2.TrainConvNet(model_save_path2)
model3.TrainConvNet(model_save_path3)


#with open("results/results.csv", 'w') as file:
#    file.write("ImageId,Label\n")
#    for idx in range(len(results1)):
#         pred1 = int(results1[idx])
#         pred2 = int(results2[idx])
#         #pred3 = int(results3[idx])
#         pred4 = int(results4[idx])
#         pred5 = int(results5[idx])
#         pred6 = int(results6[idx])
#         pred7 = int(results7[idx])

#         nums = np.zeros(10)
         
#         nums[pred1] = nums[pred1] + 1 
#         nums[pred2] = nums[pred2] + 1 
#         #nums[pred3] = nums[pred3] + 1 
#         nums[pred4] = nums[pred4] + 1 
#         nums[pred5] = nums[pred5] + 1 
#         nums[pred6] = nums[pred6] + 1 
#         nums[pred7] = nums[pred7] + 1 

#         prediction = np.argmax(nums)

#         file.write(str(idx + 1))
#         file.write(",")
#         file.write(str(prediction))
#         file.write("\n")