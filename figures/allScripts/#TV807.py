## Supplementrary video from 
## see "Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails\2021-04-15.py"

# from animal 944
# cat1 represented by touch #18  plt.xlim([-500,500])
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails\touchOnly\touchDetails/cat1.svg')
# cat2 represented by touch #36
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails\touchOnly\touchDetails/cat2.svg')
# cat3 represented by touch #2
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails\touchOnly\touchDetails/cat3.svg')
# cat4 represented by touch #60
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails\touchOnly\touchDetails/cat4.svg')


# video for pole
pvid = r"Y:\Sheldon\Highspeed\analyzed\Used_for_paper_WDIL005_Syngap1_KO_WDIL_White_noise_reduced_stim\Compressed\pole_944.avi"

t1 = allDat.loc[allDat['uniqueTchId']=='94418']
t2 = allDat.loc[allDat['uniqueTchId']=='94436']
t3 = allDat.loc[allDat['uniqueTchId']=='9442']
t4 = allDat.loc[allDat['uniqueTchId']=='94460']


def create_videoTrace(dataFrame, touchType):

	# Create a VideoWriter object
	output_path = 'output_video'+str(touchType)+'.avi'
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	fps = 3
	frame_size = (640, 480)
	out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

	# Create a Matplotlib figure
	# fig, ax = plt.subplots(frameon=False, figsize=(xdimIm / custdpi, ydimIm / custdpi))
	# Create a Matplotlib figure and axis
	# fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100)  # Adjust the figure size
	fig, ax = plt.subplots()
	ax.set_xlim(0, frame_size[0])
	ax.set_ylim(0, frame_size[1])
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove subplot margins

	# Generate frames and write them to the video
	x = np.array(dataFrame.index.to_list())
	xscale = scale_array(x, 10, 100)

	### for ploting the accel
	yacc = dataFrame['ang_accel_filt_'].values
	yacc_scale = 480-scale_array(yacc, 240, 320)

	### for ploting the curvature
	ycurv = dataFrame['filt_curvature'].values
	ycurv_scale = 480-scale_array(ycurv, 120, 200)

	### for ploting the amplitude
	yamp = dataFrame['inst_amplitude_filt'].values
	yamp_scale = 480-scale_array(yamp, 360, 420)

	for idx, frameOfInterest in enumerate(x):
	    ax.clear() # clear the previous axis

	    ## get frame from the video part
	    cap.set(1, frameOfInterest)
	    ret, frame = cap.read()
	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
	    frame = frame.astype(np.int64)
	    ax.imshow(frame, cmap='gray')
	    ax.axis('off')

	    ax.text(10, 20, 'Touch cat:  '+touchType, color='white', weight='bold', fontsize=14) # color purple
	    ax.text(10, 140, 'amplitude', color='#762a83', weight='bold') # color purple
	    ax.plot(xscale[:idx], yamp_scale[:idx], '#762a83', lw=2, label='Line')
	    ax.text(10, 260, 'acceleration', color='#1b7837', weight='bold') # color green
	    ax.plot(xscale[:idx], yacc_scale[:idx], '#1b7837', lw=2, label='Line')
	    ax.text(10, 380, 'curvature', color='#053061', weight='bold') # color blue
	    ax.plot(xscale[:idx], ycurv_scale[:idx], '#053061', lw=2, label='Line')

	    ## set the axis so that the oriation is proper
	    ax.set_xlim([0,640])
	    ax.set_ylim([480,0])

	    ### draw the canvas and maniputate the image
	    fig.canvas.draw()
	    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	    image = image[:, :, ::-1] ## convert the image back from bgr to rgb
	    
	    out.write(image)

	# Release the VideoWriter and close the Matplotlib window
	out.release()
	plt.close('all')

for i, j in tqdm.tqdm(zip([t1,t2,t3,t4],['1','2a','2b','2c'])):
	print(j)
	create_videoTrace(i,j)