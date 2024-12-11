tools<-c("ggplot2","reshape2","R.matlab","tools","base","plyr","installr")

library(ggplot2)
library(reshape2)
library(R.matlab)
library(tools)
library(base)
library(plyr)
library(installr)
library(signal)

updateR()
# theme ####
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
                  panel.background = element_rect(fill="transparent",colour=NA),
                  plot.background = element_rect(fill = "transparent", color = NA),
                  legend.key = element_rect(fill = "transparent", colour = "transparent"))

# set space ####
dir<-setwd('X:/Vaissiere/00-ephys temp/')
dir<-setwd('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/MATLA')

# find all the matlab file within the ephys temp
dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/MATLA';for(i in dir){
  setwd(i)
  setwd(i)
}
files<-list.files(pattern = '.mat', recursive = TRUE)
files<-files[-grep(".pdf", files)]
ddply()

# look for the correpsonpding table with experiment description
#exp.d<-read.csv('X:/Vaissiere/00-ephys temp/Description.csv')
exp.d<-read.csv('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Description.csv')
exp.d$files<-paste("MATLA/",
      exp.d$month,"/",
      formatC(exp.d$day, width = 2, flag="0"),"/",
      "experiment", formatC(exp.d$exp, width = 3, flag="0"), "trial",formatC(exp.d$trial, width = 3, flag="0"),".mat",
      sep = "")
exp.d<-exp.d[which(!exp.d$exp.status == "failed"),]
exp.d<-exp.d[-c(1:length(list.files('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/OUTPUT DATA'))),]

dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp';for( i in dir){
  setwd(i)
  setwd(i)
}

# to plot all the LIGHT traces individually ####
f.list<-exp.d[which(exp.d$stim.cat == "light"),]
for (i in 1:nrow(f.list)){
      data<-readMat(f.list[i,"files"])
      y1<-data.frame(data$inputData)
      y1<-data.frame(value=apply(y1,1,mean))
      y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
      temp<-y1-y.m1
      temp$id<-as.numeric(row.names(temp))

      #windows(width = 6.6, height = 2.8)
      quartz(width = 6.6, height = 2.8)
      #cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
      print(ggplot(temp, aes(y=value, x=id))+
              geom_line(alpha = 1, color = "#0098b9")+
              #facet_wrap( ~variable)+
              #labs(color="Laser power")+
              labs(title=paste(f.list[i,"date"],
                               "depth: ",
                               f.list[i, "surface"]-f.list[i, "loc"],
                               sep=""),
                   subtitle=paste("stimulation: ",
                                  f.list[i, "stim.cat"],
                                  " - ",
                                  f.list[i, "stim.dur.ms"], "ms",
                                  " - amplitude: ",
                                  f.list[i,"matlab.amplitude"],
                                  sep = ""),
                   
                   caption=paste("Bessel: ",  f.list[i, "bessel.filter.kHz"], "kHz", sep = ""))+
              geom_vline(xintercept = f.list[i, "stim.onset"]*10, linetype =8, color="red")+
              #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
              geom_hline(yintercept = 0, color = "black", alpha = 0.2)+
              xlab("Time (ms)")+
              scale_x_continuous(breaks=seq(0,f.list[i, "rec.dur"]*10, by = f.list[i, "rec.dur"]*10/5),
                                 labels=as.character(seq(0,f.list[i, "rec.dur"], by = f.list[i, "rec.dur"]/5)))+
              scale_y_continuous(limits=c(-0.5, 0.5),                           # Set y range
                                 breaks=-10:1000 * 0.1,
                                 expand = c(0,0))+
              ylab("LFP (mV)") +
              #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
              theme.tv)
      quartz.save(paste("OUTPUT DATA/",
                        f.list[i,"date"]," - ",
                        f.list[i, "stim.cat"], " - ",
                        i,
                        '.pdf',
                        sep=''),
                  type='pdf')
      #savePlot('spontat150um.pdf',type="pdf")
      dev.off()
    }

# to plot all the PIEZO traces individually ####
f.list<-exp.d[which(exp.d$stim.cat == "piezo"),]
for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"])
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  temp<-y1-y.m1
  temp$id<-as.numeric(row.names(temp))
  
  #windows(width = 6.6, height = 2.8)
  quartz(width = 6.6, height = 2.8)
  #cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
  print(ggplot(subset(temp, id<=1000 | id>=1110), aes(y=value, x=id))+
          geom_line(alpha = 1, color = "#0098b9")+
          #facet_wrap( ~variable)+
          #labs(color="Laser power")+
          labs(title=paste(f.list[i,"date"],
                           "depth: ",
                           f.list[i, "surface"]-f.list[i, "loc"],
                           sep=""),
               subtitle=paste("stimulation: ",
                              f.list[i, "stim.cat"],
                              " - ",
                              f.list[i, "stim.dur.ms"], "ms",
                              " - amplitude: ",
                              f.list[i,"matlab.amplitude"],
                              sep = ""),
               
               caption=paste("Bessel: ",  f.list[i, "bessel.filter.kHz"], "kHz", sep = ""))+
          geom_vline(xintercept = f.list[i, "stim.onset"]*10, linetype =8, color="red")+
          #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
          geom_hline(yintercept = 0, color = "black", alpha = 0.2)+
          xlab("Time (ms)")+
          scale_x_continuous(breaks=seq(0,f.list[i, "rec.dur"]*10, by = f.list[i, "rec.dur"]*10/5),
                             labels=as.character(seq(0,f.list[i, "rec.dur"], by = f.list[i, "rec.dur"]/5)))+
          scale_y_continuous(limits=c(-0.5, 0.5),                           # Set y range
                             breaks=-10:1000 * 0.1,
                             expand = c(0,0))+
          ylab("LFP (mV)") +
          #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
          theme.tv)
  quartz.save(paste("OUTPUT DATA/",
                    f.list[i,"date"]," - ",
                    f.list[i, "stim.cat"], " - ",
                    i,
                    '.pdf',
                    sep=''),
              type='pdf')
  #savePlot('spontat150um.pdf',type="pdf")
  dev.off()
}

# to plot all the spontaneous traces individually ####
f.list<-exp.d[which(exp.d$stim.cat == "spont"),]
for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"])
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  temp<-y1-y.m1
  temp$id<-as.numeric(row.names(temp))
  
  #windows(width = 6.6, height = 2.8)
  quartz(width = 6.6, height = 2.8)
  #cbPalette<-c("#abdde8","#7fcbdc","#0098b9","#d5d5d5")
  print(ggplot(temp, aes(y=value, x=id))+
          geom_line(alpha = 1, color = "#0098b9")+
          #facet_wrap( ~variable)+
          #labs(color="Laser power")+
          labs(title=paste(f.list[i,"date"],
                           "depth: ",
                           f.list[i, "surface"]-f.list[i, "loc"],
                           sep=""),
               subtitle=paste("stimulation: ",
                              f.list[i, "stim.cat"],
                              " - ",
                              f.list[i, "stim.dur.ms"], "ms",
                              " - amplitude: ",
                              f.list[i,"matlab.amplitude"],
                              sep = ""),
              caption=paste("Bessel: ",  f.list[i, "bessel.filter.kHz"], "kHz", sep = ""))+
          geom_vline(xintercept = f.list[i, "stim.onset"]*10, linetype =8, color="red")+
          #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
          geom_hline(yintercept = 0, color = "black", alpha = 0.2)+
          xlab("Time (s)")+
          scale_x_continuous(breaks=seq(0,f.list[i, "rec.dur"]*10, by = f.list[i, "rec.dur"]*10/6),
                              labels=as.character(seq(0,f.list[i, "rec.dur"]/1000, by = f.list[i, "rec.dur"]/1000/6)))+
          #scale_y_continuous(limits=c(-0.5, 0.5),                           # Set y range
          #                   breaks=-10:1000 * 0.1,
          #                   expand = c(0,0))+
          ylab("LFP (mV)") +
          #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
          theme.tv)
  quartz.save(paste("OUTPUT DATA/",
                    f.list[i,"date"]," - ",
                    f.list[i, "stim.cat"], " - ",
                    i,
                    '.pdf',
                    sep=''),
              type='pdf')
  #savePlot('spontat150um.pdf',type="pdf")
  dev.off()
}


### to plot mulitple light stim on the same graph different basis #####
exp.d<-read.csv('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Description.csv')
exp.d$files<-paste("MATLA/",
                   exp.d$month,"/",
                   formatC(exp.d$day, width = 2, flag="0"),"/",
                   "experiment", formatC(exp.d$exp, width = 3, flag="0"), "trial",formatC(exp.d$trial, width = 3, flag="0"),".mat",
                   sep = "")

exp.d<-exp.d[-c(1:length(list.files('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/OUTPUT DATA'))),]
dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp';for( i in dir){
  setwd(i)
  setwd(i)
}





# to plot all the LIGHT traces combined ####
# lapply draft to generate automation
# d <- data.frame(var1=rnorm(10), 
#                var2=rnorm(10), 
#                group=sample(c(1:3), 10, replace=TRUE))
# lapply(split(d,d$group), function(df) t.test(x=df$var1,y=df$var2))
# lapply(split(d,d$group), function(df) data.frame(df))

###### graph for the different light duration ####
f.list<-exp.d[which(exp.d$stim.cat == "light"),]
f.list<-f.list[which(f.list$group == 9),]
f.list<-f.list[1:5,]


temp<-data.frame(seq(1:(as.numeric(unique(f.list["rec.dur"]))*10)))
for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}

colnames(temp)<-c("id", as.character(unlist(f.list["stim.pulses"], use.names = FALSE)))
temp<-melt(temp, id="id")
temp$variable<-as.factor(as.numeric(as.character(temp$variable)))

quartz(width = 6.6, height = 2.8)
print(ggplot(temp, aes(y=value, x=id, group=variable, color=variable))+ #subset(temp, id>=0 & id<=4000)
        geom_line(alpha = 1)+
        #facet_wrap( ~variable)+
        #labs(color="Laser power")+
        labs(title="Motor cortex stim at 40 mW/mm2",
             subtitle=paste("recording S1",  as.numeric(unique(f.list["surface"])) - as.numeric(unique(f.list["loc"])), "um", sep=""))+
        geom_vline(xintercept = as.numeric(unique(f.list["stim.onset"]))*10, linetype =8, color="red")+
        #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        xlab("Time (ms)")+
        scale_color_discrete(name = "stim duration \n(ms)")+
        scale_x_continuous(breaks=seq(0,as.numeric(unique(f.list["rec.dur"]))*10, by = as.numeric(unique(f.list["rec.dur"]))*10/5),
                           labels=as.character(seq(0,as.numeric(unique(f.list["rec.dur"])), by = as.numeric(unique(f.list["rec.dur"]))/5)))+
        #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
        #                   breaks=-10:1000 * 0.05,
        #                   expand = c(0,0))+
        ylab("LFP (mV)") +
        #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
        theme.tv)


quartz.save(paste("OUTPUT DATA/",
                  "SUMMARY",
                  unique(as.character(unlist(f.list["date"], use.names = F)))," - ",
                  "stim.cat",
                  '1.pdf',
                  sep=''),
            type='pdf')

###### graph for the different light power 1 ms ####
f.list<-exp.d[which(exp.d$stim.cat == "light"),]
f.list<-exp.d[which(exp.d$group == 6),]
f.list<-f.list[1:5,]


temp<-data.frame(seq(1:(as.numeric(unique(f.list["rec.dur"]))*10)))
for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}
colnames(temp)<-c("id", "10","18", "25", "33", "40")
temp<-melt(temp, id="id")
temp$variable<-as.factor(as.numeric(as.character(temp$variable)))

a<-function(variable.input,scale=c(5,50), to.graph = temp){
quartz(width = 6.6, height = 2.8)
print(ggplot(to.graph, aes(y=value, x=id, group=variable, color=variable))+ #subset(temp, id>=0 & id<=4000)
        geom_line(alpha = 1)+
        #facet_wrap( ~variable)+
        #labs(color="Laser power")+
        labs(title=paste("Motor cortex stim for ", unique(unlist(f.list["stim.dur.ms"], use.names = F)),  " ms", sep=""),
             subtitle=paste("recording S1",  as.numeric(unique(f.list["surface"])) - as.numeric(unique(f.list["loc"])), "um", sep=""))+
        geom_vline(xintercept = as.numeric(unique(f.list["stim.onset"]))*10, linetype =8, color="red")+
        geom_vline(xintercept = as.numeric(unique(f.list["stim.onset"]))*10+as.numeric(unique(f.list["stim.dur.ms"]))*10, linetype =8, color="red")+
        #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        xlab("Time (ms)")+
        scale_color_discrete(name = "power \n(mW/mm2)")+
        scale_x_continuous(breaks=seq(0,as.numeric(unique(f.list["rec.dur"]))*10, by = as.numeric(unique(f.list["rec.dur"]))*10/scale),
                           labels=as.character(seq(0,as.numeric(unique(f.list["rec.dur"])), by = as.numeric(unique(f.list["rec.dur"]))/scale)))+
        #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
        #                   breaks=-10:1000 * 0.05,
        #                   expand = c(0,0))+
        ylab("LFP (mV)") +
        #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
        theme.tv)}
a(variable.input="matlab.amplitude",50, subset(temp, id>=0 & id<=2000)) #subset(temp, id>=0 & id<=4000)

quartz.save(paste("OUTPUT DATA/",
                  "SUMMARY",
                  unique(as.character(unlist(f.list["date"], use.names = F)))," - ",
                  "power 10ms",
                  'depp layer2.pdf',
                  sep=''),
            type='pdf')

###### graph for the different depth ####
f.list<-exp.d[which(exp.d$stim.cat == "light"),]
f.list<-exp.d[which(exp.d$matlab.amplitude == 5),]
f.list<-f.list[c(4,5),]


temp<-data.frame(seq(1:(as.numeric(unique(f.list["rec.dur"]))*10)))
for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}
colnames(temp)<-c("id", "-550","-300")
temp<-melt(temp, id="id")
#temp$variable<-as.factor(as.numeric(as.character(temp$variable)))

a<-function(variable.input,scale=c(5,50), to.graph = temp){
  quartz(width = 6.6, height = 2.8)
  print(ggplot(to.graph, aes(y=value, x=id, group=variable, color=variable))+ #subset(temp, id>=0 & id<=4000)
          geom_line(alpha = 1)+
          #facet_wrap( ~variable)+
          #labs(color="Laser power")+
          labs(title=paste("Motor cortex stim at 40 mW/mm2 ", unique(unlist(f.list["stim.dur.ms"], use.names = F)),  " ms", sep=""),
               subtitle=paste("recording S1 different depth",  sep=""))+
          geom_vline(xintercept = as.numeric(unique(f.list["stim.onset"]))*10, linetype =8, color="red")+
          geom_vline(xintercept = as.numeric(unique(f.list["stim.onset"]))*10+as.numeric(unique(f.list["stim.dur.ms"]))*10, linetype =8, color="red")+
          #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
          geom_hline(yintercept = 0, color = "black")+
          xlab("Time (ms)")+
          scale_color_discrete(name = "recording loc (um)")+
          scale_x_continuous(breaks=seq(0,as.numeric(unique(f.list["rec.dur"]))*10, by = as.numeric(unique(f.list["rec.dur"]))*10/scale),
                             labels=as.character(seq(0,as.numeric(unique(f.list["rec.dur"])), by = as.numeric(unique(f.list["rec.dur"]))/scale)))+
          #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
          #                   breaks=-10:1000 * 0.05,
          #                   expand = c(0,0))+
          ylab("LFP (mV)") +
          #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
          theme.tv)}
a(variable.input="matlab.amplitude",50, subset(temp, id>=0 & id<=2000)) #subset(temp, id>=0 & id<=4000)

quartz.save(paste("OUTPUT DATA/",
                  "SUMMARY",
                  unique(as.character(unlist(f.list["date"], use.names = F)))," - ",
                  "power 10ms",
                  'depth comp1.pdf',
                  sep=''),
            type='pdf')


# SINGLE TRACE #####
i<-"MATLA/Oct/11/experiment001trial019.mat"

####### to normalize all the unique traces
data<-readMat(i) 
y2<-data.frame(data$inputData)
# try to apply filter before melting
# filtering traces over 30 trace leads to average time of 4 mim
bf<-butter(n=2, W=c(59/5000,61/5000), "stop")
ss<-Sys.time()
y1<-data.frame(apply(y1, 1:2, function (x) filtfilt(bf, x)))
se<-Sys.time()
head(y1)
head(y2)
y1<-data.frame(y1, avg.trace=apply(y1, 1, mean))
#data.frame(y1, mean = apply(y1[1:1000,], 2, mean))

y1$time<-as.numeric(row.names(y1))
y1<-melt(y1, id="time")
y1.m<-ddply(subset(y1, time <= 1000),
      .(variable),
      summarize,
      mean=mean(value))
output<-merge(y1, y1.m, by="variable")
output<-transform(output, 
                  norm = value - mean)
output$prop.line[output$variable == "avg.trace"]<-1
output$prop.line[!output$variable == "avg.trace"]<-0.1

quartz()
windows()
ggplot(output, aes(x=time, y=norm, group = variable, alpha = prop.line, size = prop.line))+ 
  geom_line()+
  scale_size(range = c(0.1, 0.6))+
  scale_alpha(range = c(0.1,1))+
  theme.tv
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank())+
  theme.tv

quartz()
ggplot(subset(output, variable == "avg.trace"), aes(x=time, y=norm))+ 
  geom_line()

# trial with the filtering #####
ab<-subset(output, variable == "avg.trace", norm)
plot(ab$norm, type="l")
Fs<-10000
b<-butter(n=2, W=c(59/5000,61/5000), "stop")
c<-filtfilt(b, ab$norm)
quartz()
plot(c, type="l")
plot(ab$norm, type="l")+
  lines(c, col="red")

####### to identify peak evolution across experiment ####
f.list<-exp.d[which(exp.d$group == 9 |
                    exp.d$group == 8),]

peak.resp.f<-peak.resp

for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(y1, avg.trace=apply(y1, 1, mean))
  #data.frame(y1, mean = apply(y1[1:1000,], 2, mean))
  
  y1$time<-as.numeric(row.names(y1))
  y1<-melt(y1, id="time")
  y1.m<-ddply(subset(y1, time <= 1000),
              .(variable),
              summarize,
              mean=mean(value))
  output<-merge(y1, y1.m, by="variable")
  output<-transform(output, 
                    norm = value - mean)
  peak.resp<-ddply(subset(output, time >= 1000 & time <= 1200),
                   .(variable),
                   summarize,
                   mean=min(norm))
  peak.resp.f<-merge(peak.resp.f,peak.resp, by = "variable")

}

peak.resp.f<-peak.resp.f[-1,]
peak.resp.f$variable<-as.numeric(as.character(substr(peak.resp.f$variable, 2, 3)))
colnames(peak.resp.f)<-c("stim",seq(2:length(peak.resp.f)))
peak.resp.f<-data.frame(peak.resp.f, avg.trace=apply(peak.resp.f[2:length(peak.resp.f)],1,mean))
peak.resp.f<-melt(peak.resp.f, id="stim")

peak.resp.f$prop.line[peak.resp.f$variable == "avg.trace"]<-1
peak.resp.f$prop.line[!peak.resp.f$variable == "avg.trace"]<-0.9

which(peak.resp.f$prop.line == 0.1)

ggplot(peak.resp.f, aes(x=stim, y=value, group =variable, alpha = prop.line, size = prop.line))+
  geom_line()+
  scale_size(range = c(0.5,2))+
  scale_alpha(range = c(0.2,1))+
  theme.tv


# https://www.andrewheiss.com/blog/2017/08/10/exploring-minards-1812-plot-with-ggplot2/
####### plot the single normalized trace ####
quartz()
ggplot(output, aes(x=time, y=norm, color = variable, alpha = prop.line))+ #size = prop.line
  geom_line()+
  scale_alpha(range = c(0.3,1))+
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

# INPUT/OUTPUT CURVES #####
####### for peak ####### 

f.list<-exp.d[which(exp.d$group == 8),]
temp<-data.frame(seq(1:(as.numeric(unique(f.list["rec.dur"]))*10)))

for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}
colnames(temp)<-c("time", as.character(unlist(f.list["matlab.amplitude"], use.names = FALSE)))
temp<-melt(temp, id="time")
str(temp)
peak.resp<-ddply(subset(temp, time >= 1000 & time <= 1200),
                 .(variable),
                 summarize,
                 min=min(value))

quartz()
ggplot(peak.resp, aes(x=variable, y=min, group = 1)) +
  geom_point()+
  geom_smooth(se = TRUE, method = "lm", color="black")+
  theme.tv

peak.resp$variable<-as.numeric(peak.resp$variable)
fit<-lm(min ~ variable, peak.resp)
cor(peak.resp$min, peak.resp$variable); cor.test(peak.resp$min, peak.resp$variable)
fit$coefficients[1] #fit$coefficients[[1]]
fit$coefficients[2]
#qplot(variable, min, data = peak.resp, group=1, alpha =I(.3))+


####### for slope ####### 

f.list<-exp.d[which(exp.d$group == 8),]
temp<-data.frame(seq(1:(as.numeric(unique(f.list["rec.dur"]))*10)))

for (i in 1:nrow(f.list)){
  data<-readMat(f.list[i,"files"]) 
  y1<-data.frame(data$inputData)
  y1<-data.frame(value=apply(y1,1,mean))
  y.m1<-mean(y1$value[1:f.list[i, "stim.onset"]*10]) #normalize the response to the baseline response
  y1<-y1-y.m1
  temp<-cbind(temp,y1)}
colnames(temp)<-c("time", as.character(f.list[["matlab.amplitude"]]))
temp.slope<-subset(temp, time >= 1010 & time <= 1300)

slope.f<-function(x, low.m=0.2, high.m=0.8){
  (max(x)-min(x))*0.8 / (which.min(abs(x-min(x)*low.m)) - which.min(abs(x-min(x)*high.m)))
}
test.slope<-data.frame(power=seq(0:5)-1, slope=apply(temp.slope, 2, function(x) slope.f(x,0.2, 0.8)))
test.slope<-test.slope[-1,]

quartz()
ggplot(test.slope, aes(x=power, y=slope, group = 1)) +
  geom_point()+
  geom_smooth(se = TRUE, method = "lm", color="black")+
  theme.tv


temp.slope<-melt(temp.slope, id="time")
quartz()
ggplot(temp.slope, aes(x=time, y=value, group = variable)) +
  geom_hline(yintercept = min(temp.slope$value)*0.20, color="red")+
  geom_hline(yintercept = min(temp.slope$value)*0.80, color="red")+
  geom_line()+
  theme.tv


#min(temp.slope$value)*0.20
#min(temp.slope$value)*0.80
#slope.trial<-subset(temp.slope, value >= min(temp.slope$value)*0.80 & value <= min(temp.slope$value)*0.20)




# for signal filter

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3931263/
# find method to calculate the slope
# https://www.quora.com/What-is-the-equivalent-of-Excel-function-SLOPE-in-the-R-programming-language
# Slope = (Maximum-Minimum)* 80% / (90 to 10 decay time)
# check and install R package signal###
install.packages('signal')
library(signal)
??`signal-package`
# https://stackoverflow.com/questions/7105962/how-do-i-run-a-high-pass-or-low-pass-filter-on-data-points-in-r
# to smooth time series http://a-little-book-of-r-for-time-series.readthedocs.io/en/latest/src/timeseries.html 

