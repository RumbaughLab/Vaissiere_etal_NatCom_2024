number_of_cycles = 2
max_y = 40

x = 1:500
a = number_of_cycles * 2*pi/length(x)

y = max_y * sin(x*a)
noise1 = max_y * 1/10 * sin(x*a*10)

plot(x, y, type="l", col="red", ylim=range(-1.5*max_y,1.5*max_y,5))
points(x, y + noise1, col="green", pch=20)
points(x, noise1, col="yellow", pch=20)

quartz()

library(signal)

bf <- butter(2, 1/50, type="low")
b <- filter(bf, y+noise1)
points(x, b, col="black", pch=20)

bf <- butter(2, 1/25, type="high")
b <- filter(bf, y+noise1)
points(x, b, col="black", pch=20)


bf <- butter(2, 1/50, type="low")
b <- filtfilt(bf, y+noise1)
points(x, b, col="pink", pch=20)

bf <- butter(2, 1/25, type="high")
b <- filtfilt(bf, y+noise1)
points(x, b, col="red", pch=20)

#### expample based on simulink ####
#https://www.youtube.com/watch?v=r7ypfE5TQK0

# data set from buttLoop in MATLAB load openloop60hertz
a<-read.xlsx('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/buttloop.xls', sheetIndex = 2, header = FALSE)
a<-as.numeric(as.character(a$X1)); a<-na.omit(a); a<-as.vector(a)

e<-read.xlsx('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/buttloop.xls', sheetIndex = 1, header = FALSE)
e$X1
b<-seq(1:2000)
quartz()
plot(a)

Fs<-1000
t<-c((1:length(a)-1)/Fs)
quartz()

Fs<-1000
b<-butter(n=1, W=c(59/(Fs/2),61/(Fs/2)), "stop") #filterorder is 2 the within W 1 is the Nyquist frequency = 1/2 sampling rate of signal 
# if signal has 1000 observation per second Nyquist fq is 500 Hz
c<-filtfilt(b, a)
plot(t,a, type="l", col="blue")+
lines(t,c, col="red")+
lines(t,e$X1, col="green")

length(c)

?butter
#http://rug.mnhn.fr/seewave/HTML/MAN/bwfilter.html
install.packages('seewave')
require(signal)
library(seewave)
f <- 8000
a <- noisew(f=f, d=1)
## low-pass
# 1st order filter
res <- bwfilter(a, f=f, n=1, to=1500)
spectro(res, f=f)
res <- bwfilter(a, f=f, n=2, to=1500)
# 2nd order filter
spectro(res,f=f)
# 8th order filter
res <- bwfilter(a, f=f, n=8, to=1500)
spectro(res,f=f)
## high-pass
res <- bwfilter(a, f=f, from=2500)
spectro(res,f=f)
## band-pass
res <- bwfilter(a, f=f, from=1000, to=2000)
spectro(res,f=f)
## band-stop
res <- bwfilter(a, f=f, from=1000, to=2000,bandpass=FALSE)
spectro(res,f=f)
