## Ownership: Gerben Beintema
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

def strip_plotter(test, test_p, norm, to_img=None, plot_image=None, semi_log_y = False, n_plots=5, off_set=0, \
                  f_s=None, filename='UD-comparison-figure', cmap=None):
    DT = f_s == None
    f_s = 1 if f_s == None else f_s
    NRMS_time = np.mean(np.mean((test.y - test_p.y)**2,axis=(2,3))**0.5/norm.ystd[:,0,0],axis=1)

    def plot(ax, image):
#         image = np.transpose(np.clip(np.round(image),0,255).astype(np.uint8),(1,2,0))
        if plot_image:
            plot_image(ax, image)
        else:
            img = to_img(image)
            ax.imshow(img,cmap=cmap)
        ax.tick_params(axis='x', labelbottom=False, bottom=False)
        ax.tick_params(axis='y', labelleft=False, left=False) 

    scale_factor = 1.7
    fig = plt.figure(figsize=(scale_factor*4.96063, scale_factor*2.5),dpi=300/scale_factor)
    
    gs = GridSpec(3, n_plots, height_ratios=[1,1,0.5])
#     fig = plt.figure(figsize=(width_target/my_dpi,height_target/my_dpi),dpi=my_dpi)

    
    A = np.linspace(0,len(NRMS_time), n_plots+1)
    time_ids = ((A[1:] + A[:-1])/2).astype(int) + off_set
    for time_id, col_id in zip(time_ids,range(n_plots)):
        ax = fig.add_subplot(gs[0,col_id]) #plt.subplot2grid((3, n_plots), (0, col_id)) #real
        plot(ax, test.y[time_id])
        if col_id==0:
            ax.set_ylabel('System')
        ax = fig.add_subplot(gs[1,col_id])# plt.subplot2grid((3, n_plots), (1, col_id)) #pred
        plot(ax, test_p.y[time_id])
        if col_id==0:
            ax.set_ylabel('CNN Encoder')

    ax = fig.add_subplot(gs[2,:])#plt.subplot2grid((3,n_plots),(2,0), colspan=n_plots )
    ax.plot(np.arange(test_p.cheat_n,len(NRMS_time))/f_s, NRMS_time[test_p.cheat_n:])
    ax.semilogy()
    ax.set_xlim(0,len(NRMS_time)/f_s)
    ax.grid()
    ax.plot(time_ids/f_s, NRMS_time[time_ids],'or')
    ax.set_xlabel('Time (seconds)' if not DT else 'Time index')
    ax.set_ylabel('NRMS')
    ax.legend(['NRMS test','Times shown above'], loc='upper right')
    plt.tight_layout(pad=0.5)
    plt.savefig(filename)
    plt.show()

    
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

def make_video(test, test_p, norm, to_img, target_fps=30, f_s = None, filename = 'movies/UD-test.mp4', cmap=None):
    RMS_time = np.mean((test.y - test_p.y)**2,axis=(2,3))**0.5

    NRMS_time = np.mean(RMS_time/norm.ystd[:,0,0],axis=1) if norm is not None else RMS_time
    DT = f_s == None
    f_s = 1 if f_s == None else f_s
    
    
    L = len(test)
#     to_img = lambda y: np.transpose(np.clip(np.round(y),0,255).astype(np.uint8),(1,2,0))
    width_target = 1920
    height_target = 1080
    my_dpi = 200
    gs = GridSpec(2,2, height_ratios=[4,1])
    fig = plt.figure(figsize=(width_target/my_dpi,height_target/my_dpi),dpi=my_dpi)


    ax_system = fig.add_subplot(gs[0,0])
    ax_system.set_title('System')
    img_sys = ax_system.imshow(to_img(test.y[0]), cmap=cmap)
    # ax_system.tick_params(axis='x', labelbottom=False, bottom=False)
    # ax_system.tick_params(axis='y', labelleft=False, left=False) 

    ax_model = fig.add_subplot(gs[0,1])
    ax_model.set_title('CNN encoder')
    img_model = ax_model.imshow(to_img(test_p.y[0]), cmap=cmap)
    # ax_model.tick_params(axis='x', labelbottom=False, bottom=False)
    # ax_model.tick_params(axis='y', labelleft=False, left=False) 

    ax_error = fig.add_subplot(gs[1,:])
    ax_error.plot(np.arange(len(NRMS_time))/f_s, NRMS_time)
    ax_error.set_xlim(0,len(NRMS_time)/f_s)
    ax_error.grid()
    cur_time_plot = ax_error.plot([0/f_s], [NRMS_time[0]],'or')[0]
    ax_error.set_xlabel('Time (seconds)' if not DT else 'Time index')
    ax_error.set_ylabel('NRMS')
    ax_error.legend(['NRMS test','Current time shown'], loc='upper right')
#     plt.show()
    plt.tight_layout(pad=0.75)


    def init():
        return img_sys, img_model, cur_time_plot

    def update(i):
        img_sys.set_data(to_img(test.y[i]))
        img_model.set_data(to_img(test_p.y[i]))
        cur_time_plot.set_data([i/f_s],[NRMS_time[i]])
        return img_sys, img_model, cur_time_plot

    ani = FuncAnimation(fig, update, frames=tqdm(range(len(test.y))),
                        init_func=init, blit=True, interval=1000/target_fps)
    ani.save(filename)
    plt.close()
    from IPython.display import Video
    Video(filename)
