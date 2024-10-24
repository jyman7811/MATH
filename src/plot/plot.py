import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


data = pd.read_csv("dataset\\archive\\data\\GP21_0.6_marker.csv", sep=',')
print(data)

L_FCCx = data['L_FCC_x']
L_FCCy = data['L_FCC_y']
L_FCCz = data['L_FCC_z']

L_FM1x = data['L_FM1_x']
L_FM1y = data['L_FM1_y']
L_FM1z = data['L_FM1_z']

L_FM2x = data['L_FM2_x']
L_FM2y = data['L_FM2_y']
L_FM2z = data['L_FM2_z']

L_FM5x = data['L_FM5_x']
L_FM5y = data['L_FM5_y']
L_FM5z = data['L_FM5_z']


R_FCCx = data['R_FCC_x']
R_FCCy = data['R_FCC_y']
R_FCCz = data['R_FCC_z']

R_FM1x = data['R_FM1_x']
R_FM1y = data['R_FM1_y']
R_FM1z = data['R_FM1_z']

R_FM2x = data['R_FM2_x']
R_FM2y = data['R_FM2_y']
R_FM2z = data['R_FM2_z']

R_FM5x = data['R_FM5_x']
R_FM5y = data['R_FM5_y']
R_FM5z = data['R_FM5_z']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1)
ax.set_zlim(-0.25, 0.75)


L_FCC_SCT = ax.scatter(L_FCCx[0], L_FCCy[0], L_FCCz[0], c='b', marker='o')
L_FM1_SCT = ax.scatter(L_FM1x[0], L_FM1y[0], L_FM1z[0], c='b', marker='o')
L_FM2_SCT = ax.scatter(L_FM2x[0], L_FM2y[0], L_FM2z[0], c='b', marker='o')
L_FM5_SCT = ax.scatter(L_FM5x[0], L_FM5y[0], L_FM5z[0], c='b', marker='o')


R_FCC_SCT = ax.scatter(R_FCCx[0], R_FCCy[0], R_FCCz[0], c='r', marker='o')
R_FM1_SCT = ax.scatter(R_FM1x[0], R_FM1y[0], R_FM1z[0], c='r', marker='o')
R_FM2_SCT = ax.scatter(R_FM2x[0], R_FM2y[0], R_FM2z[0], c='r', marker='o')
R_FM5_SCT = ax.scatter(R_FM5x[0], R_FM5y[0], R_FM5z[0], c='r', marker='o')


ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('Gait Phase Visual')


ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Index', 1, 12002, valinit=1)


def update(val):
    val = round(val)

    L_FCC_SCT._offsets3d = (L_FCCx[val:val+1], L_FCCy[val:val+1], L_FCCz[val:val+1])
    L_FM1_SCT._offsets3d = (L_FM1x[val:val+1], L_FM1y[val:val+1], L_FM1z[val:val+1])
    L_FM2_SCT._offsets3d = (L_FM2x[val:val+1], L_FM2y[val:val+1], L_FM2z[val:val+1])
    L_FM5_SCT._offsets3d = (L_FM5x[val:val+1], L_FM5y[val:val+1], L_FM5z[val:val+1])
    

    R_FCC_SCT._offsets3d = (R_FCCx[val:val+1], R_FCCy[val:val+1], R_FCCz[val:val+1])
    R_FM1_SCT._offsets3d = (R_FM1x[val:val+1], R_FM1y[val:val+1], R_FM1z[val:val+1])
    R_FM2_SCT._offsets3d = (R_FM2x[val:val+1], R_FM2y[val:val+1], R_FM2z[val:val+1])
    R_FM5_SCT._offsets3d = (R_FM5x[val:val+1], R_FM5y[val:val+1], R_FM5z[val:val+1])

    fig.canvas.draw_idle()

slider.on_changed(update)


def animate(frame):
    new_val = frame * 1
    slider.set_val(new_val)


frames = 12002
interval = 1000 / 200
ani = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)

plt.show()
