# data test ####

data.subset<-exp.d[exp.d$experiment=='F0010' &
                   exp.d$stim.cat=='light' & 
                   exp.d$matlab.amplitude == 0 & 
                   exp.d$surface-exp.d$loc == -800,];data.subset

sID.list<-as.character(unique(exp.d[which(exp.d$experiment=='F0010'),'sID']))
sID.listExcluded<-c('','123','766','767','782')
sID.list<-sID.list[!sID.list %in% sID.listExcluded]

data.subset1<-data.subset[data.subset$sID %in% sID.list,]
viv.dat<-viv.final[,.(sID=Sample, geno=geno)][]
data.subset1<-merge(data.subset1,viv.dat, by='sID')


#obtainDataFromGroupLFP<-function(){
        
        temp<<-NULL
        temp<-list()
        
        f.list<-data.subset1
        
        for (i in 1:nrow(f.list)){
                
                data<-readMat(data.subset1[i,"files"]) #data<-readMat(f.list[i,"files"]) 
                y1<-data.table(data$inputData)
                y1<-melt(y1)
                y1<-y1[, timeMs:=as.numeric(row.names(y1))]
                y1<-y1[1.2e+6:2.4e+6,2:3]
                
                y1.m<-mean(y1$value)
                y1<-y1[, `:=` (value=y1[,value]-y1.m)]
                
                testseq<-seq(1,3e+6,10000/500) #for 500 Hz sampling
                y2<-y1[timeMs %in% testseq,]
                ggplot(subset(y2, timeMs %in% c(1:3000000)), aes(x=timeMs, y=value, group=1))+
                        geom_line()+
                        theme(axis.title.x=element_blank(),
                              axis.text.x=element_blank(),
                              axis.ticks.x=element_blank())
                
                
                T<-120 # second of collected data
                dt<- 1/500 #interval between time series 1/(sampling reate = 10000Hz)
                n<- T/dt #
                t<-seq(0,T,by=dt)
                f <- 1:length(t)/T
                
                testseq<-seq(1.20e+6,2.4e+6,10000/500) #for 1000 Hz sampling
                y2<-y1[timeMs %in% testseq,]
                x<-y2[,value]
                
                Y<-fft(x)
                mag <- sqrt(Re(Y)^2+Im(Y)^2)*2/n
                #phase <- atan(Im(Y)/Re(Y))
                magf<-data.table(mag[1:length(f)/2])
                magf[, `:=`(xplot=f[1:length(f)/2],
                            mag_seq=as.numeric(rownames(y1)),
                            sID=as.character(f.list[i, "sID"]))]
                temp[[i]]<-magf
                
                #print(paste('estimated time: ', seconds_to_period(1.4*nrow(f.list)+1.4-1.4*i)))
                print(paste('progress: ', i, '/', nrow(f.list)))
        }
        
        temp<-rbindlist(temp)
                temp<<-temp
                print(temp)
        
        viv.dat$sID<-as.character(viv.dat$sID)
        temp<-merge(temp, viv.dat, by='sID') 

        temp$cat[temp$xplot<=140]<-'epsilon'
        temp$cat[temp$xplot<=90]<-'gamma.h'
        temp$cat[temp$xplot<=50]<-'gamma.l'
        temp$cat[temp$xplot<=30]<-'beta'
        temp$cat[temp$xplot<=12]<-'alpha'
        temp$cat[temp$xplot<=8]<-'theta'
        temp$cat[temp$xplot<=4]<-'delta'
        temp$cat[temp$xplot<=2 | temp$xplot>=140]<-'nd'
        
        
        tempSummary<-temp[, .(sum=sum(V1), mean=mean(V1), AUC=sum(diff(xplot)*rollmeanr(V1,2))), by=c('sID','geno','cat')] 
        temp1<-subset(temp, !cat=='nd')
        tempSummary<-subset(tempSummary, !cat=='nd')
        temptot<-temp1[, .(totpower=mean(V1)), by=c('sID','geno')] 
        
        tempSummary<-merge(tempSummary, temptot, by=c('sID','geno'))
        tempSummary<-tempSummary[, .(percent=mean/totpower), by=c('sID','geno','cat')]
        

unique(tempSummary$cat)

# graphing         
ggplot(temp, aes(x=xplot, y=V1, color=geno))+
        geom_line()+
        facet_wrap(~geno)+
        ylim(c(0,0.005))+
        xlim(c(59,61))

ggplot(temp, aes(y = V1, x = xplot, colour = geno, group = geno))+
        stat_summary(fun.y = mean, geom = "line", size = 0.25) +
        xlim(c(59,62))+
        ylim(c(0,0.1))
        

tempSummary$geno<-factor(tempSummary$geno, levels=c('wt','het'))
tempSummary$cat<-factor(tempSummary$cat, levels=c('delta','theta','alpha','beta','gammma.l','gamma.h','epsilon'))

unique(tempSummary$cat)
quartz()
ggplot(tempSummary, aes(y=percent, x=geno, color=geno))+
        facet_wrap(~ cat, nrow=1)+
        geom_bar(aes(fill=geno),
                 alpha=0.8,
                 stat="summary",
                 width=1,
                 fun.y =mean,
                 color="black",
                 position=position_dodge(01))+
        #scale_fill_manual(values=cbPalette)+
        
        geom_point(
                #aes(fill=c('white')),
                color="black",
                shape=21,
                fill="grey90",
                size=2.1,
                alpha=1,
                position=position_jitter(w=0.1, h=0)) + # size set the size of the point # alpha is setting the transparency of the point
        
        stat_summary(fun.data = give.n,
                     color="white",
                     geom="text",
                     size=4)+
        #stat_summary(fun.y=mean, geom="point", shape=21, color = "black", fill="black", size=2.5) +        
        stat_summary(fun.ymin=function(x)(mean(x)-sd(x)/sqrt(length(x))),
                     fun.ymax=function(x)(mean(x)+sd(x)/sqrt(length(x))),
                     geom="errorbar", width=0.25, size=0.6, color="black")+
        annotate("text")+
        #geom_rangeframe(data=data.frame(y=c(0, 100)), aes(y)) + 
        #theme_bw() +
        #scale_y_continuous(limits = c(0, 100)) +
        #xlab("") +
        #scale_x_discrete(lables=xlabs,
        #                limits = c(1,12),
        #                breaks = 0:20 * 2)
        #                 )+
        xlab("")+
        ylab("Avg Power for fq range") +
        ggtitle("") +
        #scale_y_continuous(limits=c(0, 1.5),                           # Set y range
        #                   breaks=0:1000 * 0.5,
        #                   expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = 14, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                #axis.line.x = element_blank(),
                axis.text = element_text(size=18, color = "black"),
                axis.title = element_text(size = 18, color = "black"),
                axis.text.x = element_blank(),
                axis.ticks.x = element_blank(),
                #axis.title.x = element_text(margin=margin(0,0,0,0))
                #axis.ticks.x = element_blank(),
                axis.ticks = element_line(color="black", size=0.5),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                legend.position="NONE",
                panel.background = element_blank()
        )

t.test(tempSummary[cat=='beta',mean] ~ tempSummary[cat=='beta',geno])

# frequency range based on Goffin et al 2014
# δ, 2–4 Hz             - deep sleep
# θ, 4–8 Hz             - drowsiness
# α, 8–12 Hz            - relaxed but alert
# β, 12–30 Hz           - high alert and focused
# γlow, 30–50 Hz        - higher order cognition??
# γhigh, 50–90 Hz       - higher order cognition??
# ε, 90–140 Hz          - higher order cognition??


### test ####


plot(f[1:length(f)/2], mag[1:length(f)/2],type='l',
     col='blue',
     xlim=c(0,100),
     ylim=c(0,0.015),
     xlab='frequency',
     ylab='power')



plot(mag)
plot(mag[1:length(f)/2])
quartz()
###### EXAMPLE2 #######
library(seewave)

## Not run: 
data(tico)
data(pellucens)
# simple plots
spectro(tico,f=22050)
spectro(tico,f=22050,osc=TRUE)
spectro(tico,f=22050,scale=FALSE)
spectro(tico,f=22050,osc=TRUE,scale=FALSE)
# change the dB scale by setting a different dB reference value (20microPa)
spectro(tico,f=22050, dBref=2*10e-5)
# unnormalised spectrogram with a linear amplitude scale
spectro(tico, dB=NULL, norm=FALSE, scale=FALSE)
# manipulating wl
op<-par(mfrow=c(2,2))
spectro(tico,f=22050,wl=256,scale=FALSE)
title("wl = 256")
spectro(tico,f=22050,wl=512,scale=FALSE)
title("wl = 512")
spectro(tico,f=22050,wl=1024,scale=FALSE)
title("wl = 1024")
spectro(tico,f=22050,wl=4096,scale=FALSE)
title("wl = 4096")
par(op)
# vertical zoom using flim
spectro(tico,f=22050, flim=c(2,6))
spectro(tico,f=22050, flimd=c(2,6))
# a full plot
pellu2<-cutw(pellucens,f=22050,from=1,plot=FALSE)
spectro(pellu2,f=22050,ovlp=85,zp=16,osc=TRUE,
        cont=TRUE,contlevels=seq(-30,0,20),colcont="red",
        lwd=1.5,lty=2,palette=reverse.terrain.colors)
# black and white spectrogram 
spectro(pellu2,f=22050,ovlp=85,zp=16,
        palette=reverse.gray.colors.1)
# colour modifications
data(sheep)
spectro(sheep,f=8000,palette=temp.colors,collevels=seq(-115,0,1))
spectro(pellu2,f=22050,ovlp=85,zp=16,
        palette=reverse.cm.colors,osc=TRUE,colwave="orchid1") 
spectro(pellu2,f=22050,ovlp=85,zp=16,osc=TRUE,palette=reverse.heat.colors,
        colbg="black",colgrid="white", colwave="white",colaxis="white",collab="white")

## End(Not run)

str(tico)

## Not run: 
require(rgl)
data(tico)
spectro3D(tico,f=22050,wl=512,ovlp=75,zp=16,maga=4,palette=reverse.terrain.colors)
# linear amplitude scale without a normisation of the STFT matrix
# time and frequency scales need to be dramatically amplified
spectro3D(tico, norm=FALSE, dB=NULL, magt=100000, magf=100000)

## End(Not run)
###### EXAMPLE3 #####
quartz()

require(stats)
#Domain setup
T <- 5
dt <- 0.01 #s
n <- T/dt
F <-1/dt
df <- 1/T
freq<-5 #Hz
t <- seq(0,T,by=dt) #also try ts function

#CREATE OUR TIME SERIES DATA
y <- 10*sin(2*pi*freq*t) +4* sin(2*pi*20*t)
plot(t, type='l')

#CREATE OUR FREQUENCY ARRAY
f <- 1:length(t)/T

#FOURIER TRANSFORM WORK
Y <- fft(y)
mag <- sqrt(Re(Y)^2+Im(Y)^2)*2/n
phase <- atan(Im(Y)/Re(Y))
Yr <- Re(Y)
Yi <- Im(Y)

#PLOTTING
quartz()
layout(matrix(c(1,2), 2, 1, byrow = TRUE))
plot(t,y,type='l',xlim=c(0,T))
plot(f[1:length(f)/2],mag[1:length(f)/2],type='l')


#### GOOD TO WORK WITH
x = periodic.series(start.period = 166, length = 10000)
plot(x, type='l')

T<-120 # second of collected data
dt<- 1/500 #interval between time series 1/(sampling reate = 10000Hz)
n<- T/dt #
t<-seq(0,T,by=dt)
f <- 1:length(t)/T

testseq<-seq(1.20e+6,2.4e+6,10000/500) #for 1000 Hz sampling
y2<-y1[timeMs %in% testseq,]
x<-y2[,value]

Y<-fft(x)
mag <- sqrt(Re(Y)^2+Im(Y)^2)*2/n
phase <- atan(Im(Y)/Re(Y))

quartz()
#layout(matrix(seq(1,16), 8, 2, byrow = TRUE))
plot(seq(1,length(x)),x,type='l')
quartz()
plot(f[1:length(f)/2], mag[1:length(f)/2],type='l',
        col='blue',
        xlim=c(0,100),
        ylim=c(0,0.015),
        xlab='frequency',
        ylab='power')





#https://onestepafteranother.wordpress.com/signal-analysis-and-fast-fourier-transforms-in-r/
#https://www.sfn.org/~/media/SfN/Documents/Short%20Courses/2013%20Short%20Course%20II/SC2%20Kramer.ashx







































###### EXAMPLE4 ####
#install.packages('WaveletComp')
library(WaveletComp)
#example data
x = periodic.series(start.period = 20, length = 1000)
plot(x, type='l')
qyx = x + 0.2*rnorm(1000)

x=y2$value     
my.data = data.frame(x = x)
my.w = analyze.wavelet(my.data, "x",
                       loess.span = 0,
                       dt = 1, dj = 1/250,
                       lowerPeriod = 1,
                       upperPeriod = 1000,
                       make.pval = T, n.sim = 10)
wt.image(my.w, n.levels = 250,
         legend.params = list(lab = "wavelet power levels"))
reconstruct(my.w, plot.waves = F, lwd = c(1,2), legend.coords = "bottomleft")


t = seq(0,1,len=512)
w = 2 * sin(2*pi*16*t)*exp(-(t-.25)^2/.001)
w= w + sin(2*pi*64*t)*exp(-(t-.75)^2/.001)
w = ts(w,deltat=1/512)
plot(t,w,'l')
x = w

