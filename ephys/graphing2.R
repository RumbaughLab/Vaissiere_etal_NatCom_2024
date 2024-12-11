# setup #####
library(ggplot2)
library(reshape2)
library(R.matlab)
library(tools)
library(base)
#install.packages('R.matlab')
#ls('package:R.matlab')

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



# graphing the ligth stim #####


dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/MATLA/Sep/28')
dir<-setwd('X:/Vaissiere/00-ephys temp/Aug/23')
dir<-setwd('C:/Users/invivo-ephys/Documents/MATLA/2017/Sep/28')
dir(dir)
getwd()
files<-list.files(pattern = '.mat')
i=files[10]

files<-files[12:length(files)]

id<-c("id","75","35","0","4","22","52","111","171","290")

i=files[1]
temp<-data.frame(seq(1:10000))

for (i in files){
  data<-readMat(i) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(c1=apply(y1,1,mean))
  y.m1<-mean(y1$c1[1:2000]) #normalize the response to the baseline response
  y1<-y1-y.m1
  
  temp<-cbind(temp,y1)}
  
colnames(temp)<-c("id","75","35","0","4","22","52","111","171","290")
temp<-melt(temp, id="id")
temp$variable<-as.factor(as.numeric(as.character(temp$variable)))
str(temp)
  
  #windows(width = 6.6, height = 2.8)
  quartz(width = 6.6, height = 2.8)
  #cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
  print(ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+
          geom_line(alpha = 0.3)+
          facet_wrap( ~variable)+
          #labs(color="Laser power")+
          ggtitle("Motor cortex stim in mW/mm2")+
          geom_vline(xintercept = 2000, linetype =8, color="red")+
          geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
          geom_hline(yintercept = 0, color = "black")+
          xlab("Time (ms)")+
          scale_x_continuous(breaks=seq(0,10000, by = 2000),
                             labels=as.character(seq(0,1000, by = 200)))+
          scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
                             breaks=-10:1000 * 0.05,
                             expand = c(0,0))+
          ylab("LFP (mV)") +
          #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
          theme.tv)
  quartz.save('tom-light.pdf',type='pdf')
  quartz.save(paste(filename = basename(i),
                    '.pdf',
                    sep=''),
              type='pdf')
  #savePlot('spontat150um.pdf',type="pdf")
  dev.off()

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

