import matplotlib.pyplot as plt
import matplotlib.patches as test

# plt.figure(1)
fig, ax = plt.subplots(1)
I = plt.imread("m.png")
plt.imshow(I)
plt.title("dupa")
plt.axis('off')
# plt.show()
plt.imsave("mandril.png",I)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]
plt.plot(x,y,'gs',markersize = 10)

rect = test.Rectangle((250,250),100,100,fill = False, ec = 'b')
ax.add_patch(rect)

plt.show()





