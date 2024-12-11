library(ggplot2)
library(reshape2)
library(R.matlab)
library(tools)
library(base)

dir<-setwd('X:/Vaissiere/00-ephys temp/MATLA/Oct/06')
dir<-setwd('/Volumes//MillerRumbaughLab/Vaissiere/00-ephys temp')
files<-list.files(pattern = '.mat')
files



data<-readMat(files[1]) 

y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
min(y1$c1[1:5000])-max(y1$c1[1:10000])

boxplot(y1)

y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

y1

temp<-y1
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")

str(temp)
unique(temp$variable)


# to plot spontaneous activity ####
dir<-setwd('X:/Vaissiere/00-ephys temp/MATLA/Oct/06')
files<-list.files(pattern = '.mat')
files
data<-readMat(files[1]) 

y1<-data.frame(data$inputData)
temp<-y1
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)

windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  #labs(color="Laser power")+
  ggtitle("Spontaneous activity 150 um")+
  #geom_vline(xintercept = 2000, linetype =8, color="red")+
  #geom_hline(yintercept = 0, color = "black")+
  xlab("Time (ms)")+
  #scale_y_continuous(limits=c(-0.1, 0.1),                           # Set y range
  #                   breaks=-10:1000 * 0.05,
  #                   expand = c(0,0))+
  scale_x_continuous(breaks=seq(0,1200000, by = 200000),
                     labels=as.character(seq(0,120, by = 20)))+
  ylab("LFP (mV)") +
  #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
  theme.tv

# plot piezo ####
data<-readMat(files[2])
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat(files[3])
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:1000]) #normalize the response to the baseline response
y2<-y2-y.m2


temp<-cbind(y1,y2)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")

windows(width = 6.6, height = 2.8)
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  theme.tv
windows(width = 6.6, height = 2.8)
ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("Time (ms)")+
  ylab("LFP (mV)") +
  scale_x_continuous(breaks=seq(0,10000, by = 2000),
                     labels=as.character(seq(0,1000, by = 200)))+
  ggtitle("") +
  theme.tv


# to plot light stim with the 1kHz bessel ####
dir<-setwd('X:/Vaissiere/00-ephys temp/MATLA/Oct/06')
files<-list.files(pattern = '.mat')
files<-files[6:8]

temp<-data.frame(seq(1:10000))

for (i in files){
  data<-readMat(i) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(c1=apply(y1,1,mean))
  y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}

colnames(temp)<-c("id","19","25","31")
temp<-melt(temp, id="id")
temp$variable<-as.factor(as.numeric(as.character(temp$variable)))

print(ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
        geom_line(alpha = 1)+
        #facet_wrap( ~variable)+
        #labs(color="Laser power")+
        ggtitle("Motor cortex stim in mW/mm2")+
        geom_vline(xintercept = 2000, linetype =8, color="red")+
        geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        xlab("Time (ms)")+
        scale_x_continuous(breaks=seq(0,10000, by = 2000),
                           labels=as.character(seq(0,1000, by = 200)))+
        #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
        #                   breaks=-10:1000 * 0.05,
        #                   expand = c(0,0))+
        ylab("LFP (mV)") +
        #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
        theme.tv)

# to plot light stim with different bessel ####
dir<-setwd('X:/Vaissiere/00-ephys temp/MATLA/Oct/06')
files<-list.files(pattern = '.mat')
files<-files[8:10]

temp<-data.frame(seq(1:10000))

for (i in files){
  data<-readMat(i) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(c1=apply(y1,1,mean))
  y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}

tail(temp)
colnames(temp)<-c("id","1kHz","10kHz","2kHz")
temp<-melt(temp, id="id")
#temp$variable<-as.factor(as.numeric(as.character(temp$variable)))


print(ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
        geom_line(alpha = 1)+
        #facet_wrap( ~variable)+
        #labs(color="Laser power")+
        ggtitle("Motor cortex stim with different bessel fitlers")+
        geom_vline(xintercept = 2000, linetype =8, color="red")+
        geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        xlab("Time (ms)")+
        scale_x_continuous(breaks=seq(0,10000, by = 2000),
                           labels=as.character(seq(0,1000, by = 200)))+
        #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
        #                   breaks=-10:1000 * 0.05,
        #                   expand = c(0,0))+
        ylab("LFP (mV)") +
        #scale_colour_manual(values=c('black','black','black'))+
        #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
        theme.tv)
