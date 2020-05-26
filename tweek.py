import os
accuracy = os.system("cat /var/lib/jenkins/workspace/Pull_work/accuracy.txt")
x = "model.add(model.add(Conv2D(64, (3,3),activation= "relu"))"
if accuracy < 90:
	os.system("sed -i '/softmax/ i {}' /var/lib/jenkins/workspace/Pull_work/project.py".format(x))
	os.system("exit 1")
else:
	print("Nothing to be done more")
exit()