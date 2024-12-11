library(ggplot2)
library(reshape2)
library(R.matlab)
library(tools)
library(base)
install.packages('R.matlab')
ls('package:R.matlab')

theme.tv<-  theme(strip.text.x = element_text(size = 14, colour = "black", angle = 0),
                  strip.background = element_rect(fill="grey85", colour="black"),
                  axis.line = element_line(color="black", size=0.5),
                  #axis.line.x = element_blank(),
                  axis.text = element_text(size=12, color = "black"),
                  axis.title = element_text(size = 12, color = "black"),
                  #axis.text.x = element_blank(),
                  #axis.ticks.x = element_blank(),
                  #axis.title.x = element_text(margin=margin(0,0,0,0))
                  #axis.ticks.x = element_blank(),
                  axis.ticks = element_line(color="black", size=0.5),
                  panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.border = element_blank(),
                  #legend.position="NONE",
                  panel.background = element_blank(),
                  plot.background = element_rect(fill = "transparent", color = NA))





### 2017-08-17 piezo ####


dir<-setwd('Y:/Vaissiere/00-ephys temp/Aug/17')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment001trial007.mat')
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment001trial009.mat')
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:1000]) #normalize the response to the baseline response
y2<-y2-y.m2


temp<-cbind(y1,y2)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")


ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  theme.tv
windows(width = 6.6, height = 2.8)
ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  ggtitle("") +
  theme.tv
savePlot('2wisk.pdf',type="pdf")

### 2017-08-17 - initial light stim post setup ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Aug/17')
dir<-setwd('Y:/Vaissiere/00-ephys temp/Aug/17')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment004trial005.mat') # control
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment004trial006.mat') #100%
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment004trial007.mat') #50%
y3<-data.frame(data$inputData)
y3<-data.frame(c3=apply(y3,1,mean))
y.m3<-mean(y3$c3[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3


temp<-cbind(y1,y2)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)


windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#d5d5d5","#0098b9","#abdde8","#adff06","#d6ff82")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  geom_vline(xintercept = 2000, linetype = 8, colour = "red")+
  #scale_y_continuous(limits=c(-0.2,0.6),                         
  #                   breaks=-2:2 * 0.20,
  #                   expand = c(0,0)) +  
  #scale_colour_manual(values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/initial-stim_withEmin.pdf',type="pdf")







str(data)






















######### to check which light stimulation was performed ####
data<-readMat('experiment004trial007.mat') # control
options(max.print = 10000)
test<-data$outputData
length(which(test[,1,2]==5))

### 2017-08-18 - piezo F0001A385 Chr2+ ####

dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/18')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment001trial004.mat')
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment001trial003.mat')
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:1000]) #normalize the response to the baseline response
y2<-y2-y.m2


temp<-cbind(y1,y2)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")


windows(width = 6.6, height = 2.8)
cbPalette<-c("grey80","#1FC7CB")
ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  scale_fill_manual(values=cbPalette)+
  scale_colour_manual(values=cbPalette)+
  ggtitle("") +
  theme.tv
savePlot('X:/Vaissiere/00-ephys temp/wisk385.pdf',type="pdf")

### 2017-08-18 - piezo control Chr2- ####

dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/18')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment003trial003.mat')
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment003trial002.mat')
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:1000]) #normalize the response to the baseline response
y2<-y2-y.m2


temp<-cbind(y1,y2)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")


windows(width = 6.6, height = 2.8)
cbPalette<-c("grey80","#1FC7CB")
ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  scale_fill_manual(values=cbPalette)+
  scale_colour_manual(values=cbPalette)+
  ggtitle("") +
  theme.tv
savePlot('X:/Vaissiere/00-ephys temp/wiskctl.pdf',type="pdf")

### 2017-08-18 - light stim F0001A385 Chr2+ ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Aug/18')
dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/18')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment001trial010.mat') # control
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment001trial006.mat') #100%
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment001trial007.mat') #50%
y3<-data.frame(data$inputData)
y3<-data.frame(c3=apply(y3,1,mean))
y.m3<-mean(y3$c3[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3

data<-readMat('experiment001trial008.mat') #33%
y4<-data.frame(data$inputData)
y4<-data.frame(c4=apply(y4,1,mean))
y.m4<-mean(y4$c4[1:2000]) #normalize the response to the baseline response
y4<-y4-y.m4


temp<-cbind(y1,y2,y3,y4)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)

windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#d5d5d5","#0098b9","#7fcbdc","#abdde8")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  scale_colour_manual(values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/lightF0001A385.pdf',type="pdf")
savePlot('X:/Vaissiere/00-ephys temp/lightF0001A385.pdf',type="pdf")

### 2017-08-18 - light stim of control ####

dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/18')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment003trial009.mat') # control
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment003trial004.mat') #100%
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment003trial005.mat') #50%
y3<-data.frame(data$inputData)
y3<-data.frame(c3=apply(y3,1,mean))
y.m3<-mean(y3$c3[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3

data<-readMat('experiment003trial006.mat') #33%
y4<-data.frame(data$inputData)
y4<-data.frame(c4=apply(y4,1,mean))
y.m4<-mean(y4$c4[1:2000]) #normalize the response to the baseline response
y4<-y4-y.m4


temp<-cbind(y1,y2,y3,y4)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)

windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#d5d5d5","#0098b9","#7fcbdc","#abdde8")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  scale_y_continuous(limits=c(-0.2,0),                         
                     breaks=-0.5:100 * 0.20,
                     expand = c(0,0)) +  
  scale_colour_manual(values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/lightcontrol.pdf',type="pdf")


### 2017-08-18 - light stim of Thy1Chr2 p14 ####

dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/18')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment004trial003.mat') # control
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment004trial002.mat') #100%
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment004trial011.mat') #50%
y3<-data.frame(data$inputData)
y3<-data.frame(c3=apply(y3,1,mean))
y.m3<-mean(y3$c3[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3

data<-readMat('experiment004trial009.mat') #33%
y4<-data.frame(data$inputData)
y4<-data.frame(c4=apply(y4,1,mean))
y.m4<-mean(y4$c4[1:2000]) #normalize the response to the baseline response
y4<-y4-y.m4

data<-readMat('experiment004trial010.mat') #33%
y5<-data.frame(data$inputData)
y5<-data.frame(c5=apply(y5,1,mean))
y.m5<-mean(y5$c5[1:2000]) #normalize the response to the baseline response
y5<-y5-y.m5

temp<-cbind(y1,y2,y3,y4,y5)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)

windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#d5d5d5","#0098b9","#abdde8","#adff06","#d6ff82")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  geom_vline(xintercept = 2000, linetype = 8, colour = "red")+
  #scale_y_continuous(limits=c(-0.2,0.6),                         
  #                   breaks=-2:2 * 0.20,
  #                   expand = c(0,0)) +  
  scale_colour_manual(values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/ThyChr2.pdf',type="pdf")






























### 2017-08-23 - piezo stim #########


dir<-setwd('Y:/Vaissiere/00-ephys temp/Aug/23')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment003trial002.mat')
y1<-data.frame(data$inputData)
y1<-data.frame(c1=apply(y1,1,mean))
y.m1<-mean(y1$c1[1:1000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment003trial001.mat')
y2<-data.frame(data$inputData)
y2<-data.frame(c2=apply(y2,1,mean))
y.m2<-mean(y2$c2[1:1000]) #normalize the response to the baseline response
y2<-y2-y.m2


data<-readMat('experiment003trial004.mat') #50%
y3<-data.frame(data$inputData)
y3<-data.frame(c3=apply(y3,1,mean))
y.m3<-mean(y3$c3[1:1000]) #normalize the response to the baseline response
y3<-y3-y.m3

data<-readMat('experiment003trial003.mat') #33%
y4<-data.frame(data$inputData)
y4<-data.frame(c4=apply(y4,1,mean))
y.m4<-mean(y4$c4[1:1000]) #normalize the response to the baseline response
y4<-y4-y.m4


temp<-cbind(y1,y2,y3,y4)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")
head(temp)


windows(width = 6.6, height = 2.8)
cbPalette<-c("grey80","grey85","grey95","#1FC7CB")
ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  xlab("")+
  ylab("LFP (mV)") +
  scale_x_continuous(limits=c(0, 10000))+
  geom_vline(xintercept = 1000, linetype = 8, colour = "red")+
  scale_fill_manual(values=cbPalette)+
  scale_colour_manual(values=cbPalette)+
  
  ggtitle("") +
  theme.tv
### 2017-08-23 - light stim at 350 um deep S1 ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('Y:/Vaissiere/00-ephys temp/Aug/23')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat('experiment004trial003.mat') # 10%
y1<-data.frame(data$inputData)
y1<-data.frame(p10=apply(y1,1,mean))
y.m1<-mean(y1$p10[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment004trial004.mat') #50%
y2<-data.frame(data$inputData)
y2<-data.frame(p50=apply(y2,1,mean))
y.m2<-mean(y2$p50[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment004trial005.mat') #100%
y3<-data.frame(data$inputData)
y3<-data.frame(p100=apply(y3,1,mean))
y.m3<-mean(y3$p100[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3


temp<-cbind(y1,y2,y3)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")


windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  ggtitle("Layer2/3 - 350um deep recordings")+
  geom_vline(xintercept = 2000, linetype =8, color="red")+
  geom_hline(yintercept = 0, color = "black")+
  xlab("Time (ms)")+
  scale_x_continuous(breaks=seq(0,10000, by = 2000),
                     labels=as.character(seq(0,1000, by = 200)))+
  ylab("LFP (mV)") +
  labs(color = "Laser power")+
  scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
  theme_bw()+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/2017-08-23_lightL23.pdf',type="pdf")
savePlot('X:/Vaissiere/00-ephys temp/lightF0001A385.pdf',type="pdf")

### 2017-08-23 - light stim at 650 um deep S1 ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('C:/Users/invivo-ephys/Documents/MATLA/2017/Aug/23')
dir(dir)
files<-list.files(pattern = '.mat')


data<-readMat('experiment004trial006.mat') # 10%
y1<-data.frame(data$inputData)
y1<-data.frame(p10=apply(y1,1,mean))
y.m1<-mean(y1$p10[1:2000]) #normalize the response to the baseline response
y1<-y1-y.m1

data<-readMat('experiment004trial007.mat') #50%
y2<-data.frame(data$inputData)
y2<-data.frame(p50=apply(y2,1,mean))
y.m2<-mean(y2$p50[1:2000]) #normalize the response to the baseline response
y2<-y2-y.m2

data<-readMat('experiment004trial008.mat') #100%
y3<-data.frame(data$inputData)
y3<-data.frame(p100=apply(y3,1,mean))
y.m3<-mean(y3$p100[1:2000]) #normalize the response to the baseline response
y3<-y3-y.m3

data<-readMat('experiment004trial009.mat') #100% with 565nm
y4<-data.frame(data$inputData)
y4<-data.frame(c4=apply(y4,1,mean))
y.m4<-mean(y4$c4[1:2000]) #normalize the response to the baseline response
y4<-y4-y.m4

apply(data, FUN=function(data){mean})

temp<-cbind(y1,y2,y3,y4)
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")

str(temp)
unique(temp$variable)


windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  labs(color="Laser power")+
  ggtitle("Layer5 - 650um")+
  geom_vline(xintercept = 2000, linetype =8, color="red")+
  geom_hline(yintercept = 0, color = "black")+
  xlab("Time (ms)")+
  scale_x_continuous(breaks=seq(0,10000, by = 2000),
                   labels=as.character(seq(0,1000, by = 200)))+
  ylab("LFP (mV)") +
  scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/2017-08-23_lightL5.pdf',type="pdf")
savePlot('X:/Vaissiere/00-ephys temp/lightF0001A385.pdf',type="pdf")


### 2017-09-27 - September activity ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Sep/27')
dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('C:/Users/invivo-ephys/Documents/MATLA/2017/Sep/27')
dir(dir)
files<-list.files(pattern = '.mat')

data<-readMat("experiment002trial009.mat") 
y1<-data.frame(data$inputData)

temp<-y1
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")

str(temp)
unique(temp$variable)


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
  #scale_x_continuous(breaks=seq(0,10000, by = 2000),
  #                   labels=as.character(seq(0,1000, by = 200)))+
  ylab("LFP (mV)") +
  #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
  theme.tv
quartz.save('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/2017-08-23_lightL5.pdf',type="pdf")
savePlot('spontat150um.pdf',type="pdf")


### 2017-09-27 - September plot all traces in one go ####
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Sep/27')
dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('C:/Users/invivo-ephys/Documents/MATLA/2017/Sep/27')
dir(dir)
getwd()
files<-list.files(pattern = '.mat')
i=files[4]

for (i in files){
data<-readMat(i) 
y1<-data.frame(data$inputData)

temp<-y1
temp$id<-as.numeric(rownames(temp))
temp<-melt(temp, id="id")


#windows(width = 6.6, height = 2.8)
quartz(width = 6.6, height = 2.8)
#cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
print(ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
  geom_line()+
  #labs(color="Laser power")+
  ggtitle("Spontaneous activity 150 um")+
  #geom_vline(xintercept = 2000, linetype =8, color="red")+
  #geom_hline(yintercept = 0, color = "black")+
  xlab("Time (ms)")+
  #scale_x_continuous(breaks=seq(0,10000, by = 2000),
  #                   labels=as.character(seq(0,1000, by = 200)))+
  ylab("LFP (mV)") +
  #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
  theme.tv)
quartz.save(paste(filename = basename(i),
                  '.pdf',
                  sep=''),
            type='pdf')
#savePlot('spontat150um.pdf',type="pdf")
dev.off()
}

### trying graph automation either with for loop or lapply ####
folder<-dir
data<-readMat('experiment004trial008.mat') #100%
#data<-lapply(data, unlist, use.names=FALSE)
data.apply[[3]]

files
options(max.print = 100000)

data.apply<-lapply(files, 
       FUN=function(files){readMat(files)})
data.apply<-lapply(data.apply, unlist, use.names=FALSE, recursive = TRUE)
data.apply[[1]]
lapply(data.apply[[]][2], mean)
str(data.apply)

########### graphing example #######
ggplot(slayers, aes(y = value/sum.animal, x = as.factor(group), colour = group, group = group))+ # 
  #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
  #                                                         jitter.height = 0,
  #                                                     dodge.width = 0.6)) +
  facet_wrap(~ variable+Sex, ncol = 6)+
  geom_bar(aes(fill=group),
           alpha=0.8,
           stat="summary",
           width=1,
           fun.y = mean,
           color="black",
           position=position_dodge(01))+
  #scale_fill_manual(values=cbPalette)+
  geom_hline(yintercept = 0.50, linetype = 8, colour = "red")+
  geom_point(#aes(fill=group),
    fill="grey90",
    color="black",
    shape=21,
    size=2.1,
    alpha=1,
    position=position_jitter(w=0.1, h=0)) + # size set the size of the point # alpha is setting the transparency of the point
  stat_summary(fun.data = give.n,
               color="white",
               geom="text",
               size=5)+
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
  ylab("Starter layers proportion") +
  ggtitle("") +
  scale_y_continuous(limits=c(0,1),                           # Set y range
                     breaks=0:100 * 0.20,
                     expand = c(0,0)) +                      # Set tick every 4
  #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
  scale_fill_manual(values=cbPalette)+
  #scale_colour_manual(values=cbPalette)+
  #theme_bw()+
  theme(strip.text.x = element_text(size = 14, colour = "black", angle = 0),
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
        #legend.position="NONE",
        panel.background = element_blank(),
        plot.background = element_rect(fill = "transparent", color = NA)) # important to generate transparent output

quartz.save('/Users/lab/Ox0010_analysis/starter-layerprop.pdf', type="pdf")

