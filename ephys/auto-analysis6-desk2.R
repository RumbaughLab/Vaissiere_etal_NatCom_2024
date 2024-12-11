# note to clear the environment rm(list=ls())
# LIBRARY ####
library(ggplot2)
library(reshape2)
library(R.matlab)
library(tools)
library(base)
library(plyr)
library(data.table)
library(lubridate)
# FUNCTION - THEME ####
txtsizetv<-10
theme.tv<-  theme(strip.text.x = element_text(size = txtsizetv, colour = "black", angle = 0),
                  strip.background = element_rect(fill="grey85", colour="black"),
                  axis.line = element_line(color="black", size=0.5),
                  text = element_text(size=txtsizetv),
                  #axis.line.x = element_blank(),
                  axis.text = element_text(size=txtsizetv, color = "black"),
                  axis.title = element_text(size = txtsizetv, color = "black"),
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

##### Multiple plot function ##
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
# FUNCTION - DATA EXTRACTION ####
# function and execution to obtain trace data
# baselineTrace unit is in ms the default is 200ms
to.include<-200
i<-1
obtainDataFromGroup<-function(to.include, ntracesAvg=15, baselineTrace=200){
  
  f.list<-exp.d[exp.d$group %in% to.include,]
  temp<<-NULL
  temp<-list()
  for (i in 1:nrow(f.list)){
    data<-readMat(f.list[i,"files"]) 
    y1<-data.table(data$inputData)
    y1<-y1[,1:ntracesAvg]
    y1<-data.table(y1[, value:=rowMeans(.SD)])    # equivalent to y1<-data.table(value=apply(y1,1,mean))
    y.m1<-mean(y1[(f.list[i, "stim.onset"]*10-baselineTrace*10):(f.list[i,"stim.onset"]*10), value]) # 2000 baseline for the first 200 ms
    y1<-data.table(value=y1[0:(f.list[i, "stim.onset"]*10+f.list[i, "post.stim.ms"]*10), value]-y.m1) # go up to 500 ms poststim
    y.sd<-mean(y1[(f.list[i, "stim.onset"]*10-baselineTrace*10):(f.list[i,"stim.onset"]*10), value])-3*sd(y1[(f.list[i, "stim.onset"]*10-baselineTrace*10):(f.list[i,"stim.onset"]*10), value])
    y1[, `:=`(timeMs=as.numeric(rownames(y1)),
              baseline3SD=y.sd,
              sID=as.character(f.list[i, "sID"]),
              stimDur=as.numeric(as.character(f.list[i, "stim.dur.ms"])),
              depth=f.list[i, "surface"] - f.list[i, "loc"],
              group=f.list[i, "group"],
              date=as.character(f.list[i, "date"]),
              recDur=f.list[i, "rec.dur"],
              stimOnset=f.list[i, "stim.onset"],
              whisker=f.list[i,"whisker"],
              desired.mWmm2=f.list[i,"desired.mWmm2"],
              matlabInput=substr(as.character(f.list[i, "matlab.amplitude"]), 1,5))]
    temp[[i]]<-y1
    #print(paste('estimated time: ', seconds_to_period(1.4*nrow(f.list)+1.4-1.4*i)))
    print(paste('progress: ', i, '/', nrow(f.list)))
  }
  
  temp<-rbindlist(temp)
  if (is.na(temp$stimDur[[1]])){
    temp<<-temp
    print(temp)
  }else{
    #temp$desired.mWmm2<-round(exp.d$desired.mWmm2, digits = 1)
    #temp$desired.mWmm2<-as.character(temp$desired.mWmm2)
    temp$desired.mWmm2<-formatC(temp$desired.mWmm2, width = 2, flag = "0")
    temp<<-temp
    #fwrite(temp, 'temp.csv', row.names = FALSE)
    print(temp)
  }
}

# FUNCTION - GRAPHING ####
graphTraceMain<-function(to.graph = temp,
                         groupCluster = 18,
                         scale=c(5,50), 
                         display.begin=0,
                         display.end=500){
  if (is.na(temp$stimDur[[1]])){
    #quartz(width = 6.6, height = 2.8)
    graphTraceMainOutput<<-ggplot(subset(to.graph, timeMs<=1000 | timeMs>=1110 & timeMs>=display.begin*10 & timeMs<=display.end*10 & group==groupCluster), 
                                  aes(y=value, x=timeMs, group=whisker, color=whisker))+ #subset(temp, id>=0 & id<=4000)
      geom_line(alpha = 1)+
      #facet_wrap( ~variable)+
      #labs(color="Laser power")+
      labs(title=paste("Motor cortex - " , "sID: ", temp[group==groupCluster, sID][[1]],
                       " - stim for ", temp[group==groupCluster, stimDur][[1]],  " ms", sep=""),
           subtitle=paste("recording S1",  temp[group==groupCluster, depth][[1]], "um", sep=""),
           " - ", temp[group==groupCluster, date][[1]])+
      geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color='#35e2f2')+
      geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color='#35e2f2')+
      #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
      geom_hline(yintercept = 0, color = "black")+
      xlab("Time (ms)")+
      scale_color_discrete(name = "Whisker stim.")+
      scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                         labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
      #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
      #                   breaks=-10:1000 * 0.05,
      #                   expand = c(0,0))+
      ylab("LFP (mV)") +
      #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
      theme.tv
  }else{
    #quartz(width = 6.6, height = 2.8)
    graphTraceMainOutput<<-ggplot(subset(to.graph, timeMs>=display.begin*10 & timeMs<=display.end*10 & group==groupCluster), 
                                  aes(y=value, x=timeMs, group=desired.mWmm2, color=desired.mWmm2))+ #subset(temp, id>=0 & id<=4000)
      geom_line(alpha = 1)+
      #facet_wrap( ~variable)+
      #labs(color="Laser power")+
      labs(title=paste("Motor cortex - " , "sID: ", temp[group==groupCluster, sID][[1]],
                       " - stim for ", temp[group==groupCluster, stimDur][[1]],  " ms", sep=""),
           subtitle=paste("recording S1",  temp[group==groupCluster, depth][[1]], "um", sep=""),
           " - ", temp[group==groupCluster, date][[1]])+
      geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color='#35e2f2')+
      geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color='#35e2f2')+
      #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
      geom_hline(yintercept = 0, color = "black")+
      xlab("Time (ms)")+
      scale_color_discrete(name = "power \n(mW/mm2)")+
      scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                         labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
      #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
      #                   breaks=-10:1000 * 0.05,
      #                   expand = c(0,0))+
      ylab("LFP (mV)") +
      #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
      theme.tv
  }
}
graphTrace<-function(to.graph = temp,
                     groupCluster = 13,
                     scale=c(5,50), 
                     display.begin=0,
                     display.end=500){
  
  #quartz(width = 6.6, height = 2.8)
  graphTraceOutput<<-ggplot(subset(to.graph, timeMs>=display.begin*10 & timeMs<=display.end*10 & group==groupCluster), 
                            aes(y=value, x=timeMs, group=desired.mWmm2, color=desired.mWmm2))+ #subset(temp, id>=0 & id<=4000)
    geom_line(alpha = 1)+
    #facet_wrap( ~variable)+
    #labs(color="Laser power")+
    labs(title=paste("Motor cortex - " , "sID: ", temp[group==groupCluster, sID][[1]],
                     " - stim for ", temp[group==groupCluster, stimDur][[1]],  " ms", sep=""),
         subtitle=paste("recording S1",  temp[group==groupCluster, depth][[1]], "um", sep=""),
         " - ", temp[group==groupCluster, date][[1]])+
    geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
    geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
    #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
    geom_hline(yintercept = 0, color = "black")+
    xlab("Time (ms)")+
    scale_color_discrete(name = "power \n(mW/mm2)")+
    scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                       labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
    #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
    #                   breaks=-10:1000 * 0.05,
    #                   expand = c(0,0))+
    ylab("LFP (mV)") +
    #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
    theme.tv
}

graphPeak<-function(groupCluster = 13,
                    lowerInterval = 5010,
                    upperInterval = 5040){
  peak.resp<-temp[timeMs>=lowerInterval*10 & timeMs<=upperInterval*10 & group == groupCluster, min(value), by=desired.mWmm2]
  peak.resp$desired.mWmm2<-as.numeric(peak.resp$desired.mWmm2)  
  #quartz(width = 3.2, height = 2.8)
  graphPeakOutput<<-ggplot(peak.resp, aes(x=desired.mWmm2, y=V1, group = groupCluster)) +
    geom_smooth(se = TRUE, method = "lm", color="black")+
    labs(title="Peak amplitude")+
    geom_point(color='black')+
    xlab("Power (mW/mm2)")+
    ylab("LFP (mW)")+
    theme.tv
}

dataPeak<-function(groupCluster = 103,
                   lowerInterval = 111,
                   upperInterval = 180){
  peak.resp<<-temp[timeMs>=lowerInterval*10 & timeMs<=upperInterval*10 & group == groupCluster, 
                   .(minPeak=min(value), 
                     maxPeak=max(value),
                     minPeakTime=(lowerInterval*10+which.min(value))/10,
                     maxPeakTime=(lowerInterval*10+which.max(value))/10,
                     deltaTime=((lowerInterval*10+which.max(value))/10-(lowerInterval*10+which.min(value))/10),
                     slopeMinMax=(max(value)-min(value))/((lowerInterval*10+which.max(value))/10-(lowerInterval*10+which.min(value))/10)), 
                   by=c('group','sID','depth','desired.mWmm2','whisker')]
}


temp$row<-row.names(temp)

slopeFunction<-function(x, low.m=0.2, high.m=0.8){
  (max(x)-min(x))*high.m / (which.min(abs(x-min(x)*low.m)) - which.min(abs(x-min(x)*high.m)))
}

graphSlope<-function(groupCluster = 13,
                     lowerInterval = 5010,
                     upperInterval = 5040){
  slope<-temp[timeMs>=lowerInterval*10 & timeMs<=upperInterval*10 & group==groupCluster, slopeFunction(value), by=desired.mWmm2]
  slope$desired.mWmm2<-as.numeric(slope$desired.mWmm2)  
  #quartz(width = 3.2, height = 2.8)
  graphSlopeOutput<<-ggplot(slope, aes(x=desired.mWmm2, y=V1, group = 1)) +
    geom_smooth(se = TRUE, method = "lm", color="black")+
    geom_point(color='black')+
    labs(title="Slope 20/80")+
    xlab("Power (mW/mm2)")+
    ylab("Slope 20/80")+
    theme.tv
}

graphSummaryTPS<-function(groupCluster = 34){
  
  if (Sys.info()['sysname']=='Windows'){windows(width = 13, height = 5.6)}else{quartz(width = 13, height = 5.6)}
  graphTraceMain(to.graph = temp,
                 groupCluster = groupCluster,
                 scale=50,
                 display.begin=0, #4800, 0
                 display.end=500) #5500, 500
  
  graphTrace(to.graph = temp,
             groupCluster = groupCluster,
             scale=5000,
             display.begin=200, #5000, 200
             display.end=230) #5030, 230
  
  graphPeak(groupCluster = groupCluster,
            lowerInterval = 100,
            upperInterval = 140)
  
  
  graphSlope(groupCluster = groupCluster,
             lowerInterval = 100,
             upperInterval = 140)
  
  layout <- matrix(c(rep(1, 5), 2, 2, 2, 3, 4), nrow = 2, byrow = TRUE)
  multiplot(graphTraceMainOutput, graphTraceOutput, graphPeakOutput, graphSlopeOutput, layout = layout)
  
  if (Sys.info()['sysname']=="Windows"){
    savePlot(paste("OUTPUT DATA/",
                   "SUMMARY_", Sys.Date(),'_',
                   "sID", temp[group==groupCluster, sID][[1]], 
                   "_", temp[group==groupCluster, stimDur][[1]], "ms_",
                   "depth",temp[group==groupCluster, depth][[1]],
                   '.pdf',
                   sep=''),
             type='pdf')
  }else{
    quartz.save(paste("OUTPUT DATA/",
                      "SUMMARY_", Sys.Date(),'_',
                      "sID", temp[group==groupCluster, sID][[1]], 
                      "_", temp[group==groupCluster, stimDur][[1]], "ms_",
                      "depth",temp[group==groupCluster, depth][[1]],
                      '.pdf',
                      sep=''),
                type='pdf') 
  }
  #dev.off()
}

# FUNCTION - WHISKER ####
graphWisker<-function(groupCluster){obtainDataFromGroup(c(groupCluster), baselineTrace = 100) 
  dataPeak(groupCluster); peak.resp
  graphTraceMain(to.graph = temp,
                 groupCluster = groupCluster,
                 scale=25,
                 display.begin=000,
                 display.end=410); graphTraceMainOutput
  whisker<-read.table(text='whisker ML AP s.d.rel s.d.ab half-width
                      a 3.56 2.26 38 287 348
                      b 3.16 2.02 30 324 354
                      g 2.81 1.68 27 329 350
                      d 2.71 1.39 31 297 348
                      A1 3.77 2.15 41 359 334
                      A2 3.91 1.95 43 379 336
                      A3 4.04 1.74 32 270 179
                      B1 3.47 1.97 28 269 358
                      B2 3.62 1.79 25 323 358
                      B3 3.81 1.60 37 297 340
                      C1 3.19 1.77 21 269 340
                      C2 3.35 1.61 27 278 340
                      C3 3.47 1.47 33 280 340
                      C4 3.65 1.34 40 306 348
                      C5 3.79 1.27 42 336 414
                      D1 3.00 1.50 22 281 348
                      D2 3.13 1.33 31 275 354
                      D3 3.20 1.15 37 304 360
                      D4 3.33 1.03 34 326 366
                      D5 3.52 1.00 49 353 366
                      D6 3.67 1.01 16 461 464
                      E1 2.79 1.12 33 322 358
                      E2 2.91 0.92 35 351 364
                      E3 3.02 0.74 41 333 368
                      E4 3.18 0.63 56 284 374
                      E5 3.36 0.62 51 306 386
                      E6 3.48 0.69 60 276 530
                      target 3.5 2.0 0 0 0', header=TRUE)
  whisker<-whisker[,1:3]
  whiskerPlot<-merge(whisker, peak.resp, all=TRUE)
  #whiskerPlot<-whiskerPlot[-28,]
  whiskerPlot[is.na(whiskerPlot$V1),"V1"]<-0
  whiskerPlot$V1<-abs(whiskerPlot$V1)
  whiskerPlot$TargetArea[whiskerPlot$V1 > 0]<-"stim"
  whiskerPlot$TargetArea[whiskerPlot$V1 == 0]<-"no stim"
  whiskerPlot$TargetArea[whiskerPlot$whisker == "target"]<-"target"
  whiskerPlot<-rename(whiskerPlot, c(V1 = "Peak amp."))
  
  whiskMap<-ggplot(whiskerPlot, aes(ML, AP, label=whisker, size = `Peak amp.`, color=TargetArea))+
    geom_point()+
    geom_text(size=3,hjust = 0.5, vjust = 2.5 )+
    scale_color_manual(values = c('black',"#83c483",'#ed4b31'))+
    theme(panel.grid.major = element_line(colour = 'black'))+
    theme_bw()+
    scale_y_continuous(limits = c(0.4,2.3))
  #scale_x_continuous(limits = c(0,4))
  
  layout <- matrix(c(rep(1, 3), 2,2), nrow = 1, byrow = TRUE)
  if (Sys.info()['sysname']=='Windows'){windows(width = 12, height = 4)}else{quartz(width = 12, height = 4)}
  multiplot(graphTraceMainOutput, whiskMap, layout = layout)
  
  
  if (Sys.info()['sysname']=="Windows"){
    savePlot(paste("OUTPUT DATA/",
                   "SUMMARY_", Sys.Date(),'_',
                   "sID", temp[group==groupCluster, sID][[1]], 
                   "_", temp[group==groupCluster, stimDur][[1]], "ms_",
                   "depth",temp[group==groupCluster, depth][[1]],
                   '_WhiskMap.pdf',
                   sep=''),
             type='pdf')
  }else{
    quartz.save(paste("OUTPUT DATA/",
                      "SUMMARY_", Sys.Date(),'_',
                      "sID", temp[group==groupCluster, sID][[1]], 
                      "_", temp[group==groupCluster, stimDur][[1]], "ms_",
                      "depth",temp[group==groupCluster, depth][[1]],
                      '_WhiskMap.pdf',
                      sep=''),
                type='pdf') 
  
}
}
# copy files ####
dir.create('Y:/Vaissiere/00-ephys temp/MATLA/2018/Mar/13')
file.copy(from=list.files('C:/Users/invivo-ephys/Documents/MATLA/2018/Mar/13', full.names = TRUE), 
          to='Y:/Vaissiere/00-ephys temp/MATLA/2018/Mar/13', 
          overwrite = TRUE) 
 

# LOAD - REF FILE ####
if(Sys.info()['sysname']=='Windows'){
dir<-'Y:/Vaissiere/00-ephys temp/';for(i in dir){
        setwd(i)
        setwd(i)
}
}else{
dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp';for(i in dir){
        setwd(i)
        setwd(i)
}
}

# find all the matlab file within the ephys temp
files<-list.files(pattern = '.mat', recursive = TRUE)
files<-files[-grep(".pdf", files)]


# look for the correpsonpding table with experiment description
exp.d<-read.csv('Description.csv')
exp.d$files<-paste("MATLA/",
                   exp.d$year,"/",
                   exp.d$month,"/",
                   formatC(exp.d$day, width = 2, flag="0"),"/",
                   "experiment", formatC(exp.d$exp, width = 3, flag="0"), "trial",formatC(exp.d$trial, width = 3, flag="0"),".mat",
                   sep = "")
tail(exp.d$group)
#exp.d<-exp.d[-c(1:length(list.files('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/OUTPUT DATA'))),]


# OUTPUT   ####

# data light
dataLi<-c(415)
obtainDataFromGroup(dataLi, ntracesAvg=15, baselineTrace = 100) 
dataPeak(dataLi, 5000, 5020); peak.resp

# graph light
for (i in c(dataLi)){graphSummaryTPS(i)}

# graph whisker
graphWisker(113)


grpahunique(temp$group)
# stat slope #
fit<-lm(V1 ~ desired.mWmm2, peak.resp)
cor(peak.resp$V1, peak.resp$desired.mWmm2); cor.test(peak.resp$V1, peak.resp$desired.mWmm2)
fit$coefficients[1] #fit$coefficients[[1]]
fit$coefficients[2]
#qplot(variable, min, data = peak.resp, group=1, alpha =I(.3))+


## subset for time identical ####
unique(temp$desired.mWmm2)
unique(temp$stimDur)
formatC(temp$stimDur, width=2, flag='0')
temp1<-temp
graphTraceTime<-function(to.graph = temp,
                     groupCluster = "10",
                     scale=c(5,50), 
                     display.begin=0,
                     display.end=500){
  temp$stimDur<<-formatC(temp$stimDur, width=2, flag='0')
  #quartz(width = 6.6, height = 2.8)
  graphTraceTimeOutput<<-ggplot(subset(to.graph, timeMs>=display.begin*10 & timeMs<=display.end*10 & desired.mWmm2==groupCluster), 
                            aes(y=value, x=timeMs, group=stimDur, color=stimDur))+ #subset(temp, id>=0 & id<=4000)
    geom_line(alpha = 1)+
    #facet_wrap( ~variable)+
    #labs(color="Laser power")+
    labs(title=paste("Motor cortex - " , "sID: ", temp[desired.mWmm2==groupCluster, sID][[1]],
                     " - stim for ", temp[desired.mWmm2==groupCluster, desired.mWmm2][[1]],  " mWmm2", sep=""),
         subtitle=paste("recording S1",  temp[desired.mWmm2==groupCluster, depth][[1]], "um", sep=""),
         " - ", temp[desired.mWmm2==groupCluster, date][[1]])+
    geom_vline(xintercept = temp[desired.mWmm2==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
    #geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
    #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
    geom_hline(yintercept = 0, color = "black")+
    xlab("Time (ms)")+
    scale_color_discrete(name = "stim. (ms)")+
    scale_x_continuous(breaks=seq(0,temp[desired.mWmm2==groupCluster, recDur][[1]]*10, by = temp[desired.mWmm2==groupCluster, recDur][[1]]*10/scale),
                       labels=as.character(seq(0,temp[desired.mWmm2==groupCluster, recDur][[1]], by = temp[desired.mWmm2==groupCluster, recDur][[1]]/scale)))+
    #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
    #                   breaks=-10:1000 * 0.05,
    #                   expand = c(0,0))+
    ylab("LFP (mV)") +
    #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
    theme.tv
}
windows()
i<-"10"
graphList<-NULL
graphList<-list()
stimlist<-unique(temp$desired.mWmm2)
stimlist<-stimlist[-1]
for (i in stimlist){graphTraceTime(temp, i, 500, 4950, 5100)
  graphList[[i]]<-graphTraceTimeOutput
  }
graphList$`10`

layout <- matrix(c(seq(1, 9, 1),10), nrow = 5, byrow = TRUE)
multiplot(graphList$`01`, graphList$`05`,graphList$`15`,graphList$`20`,graphList$`25`,graphList$`30`,
          graphList$`35`,graphList$`40`,layout = layout)
quartz.save('SUMMARY_2017-11-20-scalingWithTime.pdf',type='pdf')
w
## EXTRA WITH DRUG ######
dataPeak(serie, 5010, 5040); peak.resp
test<-merge(exp.d2, peak.resp, by='group')
test<-test[!duplicated(test$V1),]
length(test$V1)


peak.resp$group[6:10]<-'01min'

fwrite(peak.resp, 'peak.resp.lightdrug.csv')
peak.resp<-fread('peak.resp.lightdrug.csv')
peak.resp$ledstim<-peak.resp$ledstim*10

color<-c('#e6aaaa','#d77676','#cd5555', 'grey')

graphPeak(47,111,180)
windows(2,2)
ggplot(subset(peak.resp, !ledstim==50), aes(x=ledstim, y=V1, group=group, color=group)) +
  geom_smooth(method = "lm", aes(fill=group))+
        geom_point()+
        labs(title="Peak amplitude CNQX/AP5")+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        xlab("Power (mW/mm2)")+
  ylab("LFP (mW)")+
  theme.tv
savePlot('peak.resp.ligthdrug.pdf',type='pdf')
slopeFunction<-function(x, low.m=0.2, high.m=0.8){
  (max(x)-min(x))*high.m / (which.min(abs(x-min(x)*low.m)) - which.min(abs(x-min(x)*high.m)))
}

graphSlope<-function(groupCluster = 44,
                     lowerInterval = 111,
                     upperInterval = 180){
  slope<<-temp[timeMs>=lowerInterval*10 & timeMs<=upperInterval*10 & group==groupCluster, slopeFunction(value), by=whisker]
  slope$desired.mWmm2<<-as.numeric(slope$desired.mWmm2)  
  #quartz(width = 3.2, height = 2.8)
  graphSlopeOutput<<-ggplot(slope, aes(x=desired.mWmm2, y=V1, group = 1)) +
    geom_smooth(se = TRUE, method = "lm", color="black")+
    geom_point(color='black')+
    labs(title="Slope 20/80")+
    xlab("Power (mW/mm2)")+
    ylab("Slope 20/80")+
    theme.tv
}
windows()
graphSlope(44,111,180)
slope$time<-c(0,0,5,10,20,30,40)
slope$type<-c('no drug', rep('CNQX/AP5',6))
graphSlope(44,100,180)
serie

## PILOT EXTRA PILOT ####


#sID.list<-c(122,121,113,114,111,112)
sID.list<-as.character(unique(exp.d[which(exp.d$experiment=='F0010'),'sID']))
sID.listExcluded<-c('','123','766','767','782')
sID.list<-sID.list[!sID.list %in% sID.listExcluded]

serie<-c(200,202,204,208)
serie<-c('200','202','204','208') # motor
serie<-c('201','203','207','210') # thal

sID.list<-as.character(unique(exp.d[which(exp.d$group %in% serie),'sID']))
serie<-unique(exp.d2$group)


exp.d1<-exp.d[exp.d$sID %in% sID.list,]
exp.d2<-exp.d1[which(exp.d1$stim.cat == 'light'),]


#obtainDataFromGroup(serie, baselineTrace = 200) 
#fwrite(temp, 'temp.csv') fwrite(temp, 'temp-thal.csv') fwrite(temp, 'temp-motor.csv')
temp<-fread('temp.csv')
temp$sID<-as.character(temp$sID)
dataPeak(serie, 5000, 5060); peak.resp

viv.final<-merge(viv, out, by=c("Sample","WellPlate"))
viv.final<-viv.final[viv.final$Sample %in% sID.list]
#viv.final<-viv.final[!duplicated(viv.final$Sample),c('Sample','geno','Sex')]
viv.final<-rename(viv.final, c(Sample = 'sID'))
viv.final$sID<-as.character(viv.final$sID)

ddply(viv.final,
      .(geno, Sex),
      summarize,
      l=length(geno))

peak.resp<-merge(viv.final, peak.resp, by='sID')
peak.resp$desired.mWmm2<-as.numeric(peak.resp$desired.mWmm2)
peak.resp$depth<-factor(peak.resp$depth, levels = c(-300,-800))
peak.resp$geno<-factor(peak.resp$geno, levels = c('wt','het'))

peak.resp[minPeak == min(peak.resp$minPeak),]
peak.respLigth<-peak.resp


quartz(,2.8,1.6)
quartz(,2,2)
windows()
color<-c('blue','red')
toplottv<-c(0,1,5)
toplottv2<-c(0)


bpeakR<-peak.resp[, .(mean=mean(minPeak), lwr=mean(minPeak)-sd(minPeak)/sqrt(length(minPeak)), upr=mean(minPeak)+sd(minPeak)/sqrt(length(minPeak))), by=c('geno','depth','desired.mWmm2')]

bpeakR$depth<-factor(bpeakR$depth, levels = c(-300,-800))
bpeakR$geno<-factor(bpeakR$geno, levels = c('wt','het'))

ggplot(subset(bpeakR, depth=='-800' & !desired.mWmm2==60), aes(x=desired.mWmm2, y=mean, fill=geno, color=geno)) +
        
        geom_line(aes(color = geno), size=1, alpha=1)+
        #facet_wrap(~depth)+
        geom_ribbon(aes(ymax=upr,ymin=lwr, fill = geno), color=NA, alpha =.3)+
        geom_point(alpha=0.5)+



        #scale_y_continuous(limits=c(-1.5, 0.25),                           # Set y range
        #                   breaks=-6:20000 * 0.5) +                      # Set tick every 4
        #labs(title="Peak amplitude - supp layer (-300 um) & deep (-800 um) layers")+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        xlab("Power (mW/mm2)")+
        ylab("LFP (mW)")+
        theme.tv+
        theme(legend.position = 'NONE')
quartz.save('fig-pilotdata_supp10.pdf', type = 'pdf')


fit<-aov(minPeak ~ geno*desired.mWmm2 + Error(sID), subset(peak.resp, !desired.mWmm2 %in% toplottv))
summary(fit)


ggplot(subset(peak.resp, !sID %in% toplottv2 & !desired.mWmm2 %in% toplottv), 
       aes(x=desired.mWmm2, y=minPeak, group=geno, color=geno)) +
        geom_smooth(method = "lm", aes(fill=geno), alpha=0.4)+
        facet_wrap(~depth)+
        geom_point(alpha=0.2)+
        geom_line(aes(group=sID, color = geno), size=1, alpha=0.2)+
        scale_y_continuous(limits=c(-1.5, 0.25),                           # Set y range
                           breaks=-6:20000 * 0.5) +                      # Set tick every 4
        labs(title="Peak amplitude - supp layer (-300 um) & deep (-800 um) layers")+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        xlab("Power (mW/mm2)")+
        ylab("LFP (mW)")+
        theme.tv
quartz.save('pilotdata_supp2.pdf', type = 'pdf')
dev.size()


## PILOT EXTRA F0020 list for F0020 pilot ####
# for id list
a<-unique(exp.d[which(exp.d$experiment == 'F0020'),'sID'])
a<-as.character(a); a;length(a)

#for old motor fast 
# no motor fast for id 755, 754, 751
# 751 motor fast but different ramp
MotFast<-c(0.1,0.5,1,2,3,4,5)
exp.d10
exp.d[which(exp.d$experiment == 'F0020' &
              exp.d$desired.mWmm2 %in% MotFast &
              exp.d$stim.site == 'motor' &
              exp.d$sID == 755),'group']
exp.d[which(exp.d$sID == a[3]),]
serie<-unique(exp.d10$group)

#for new motor fast
MotFast<-c(0.1,0.5,1,2,3,4,5)
exp.d[which(exp.d$experiment == 'F0020' &
              exp.d$desired.mWmm2 %in% MotFast &
              exp.d$stim.site == 'motor new coord'),]


## PILOT EXTRA F0020 ####


#sID.list<-c(122,121,113,114,111,112)
sID.list<-as.character(unique(exp.d[which(exp.d$experiment=='F0020'),'sID']))


serie<-c(200,202,204,208)
serie<-c('200','202','204','208') # motor
serie<-c('201','203','207','210') # thal

sID.list<-as.character(unique(exp.d[which(exp.d$group %in% serie),'sID']))
serie<-unique(exp.d2$group)


exp.d1<-exp.d[exp.d$sID %in% sID.list,]
exp.d2<-exp.d1[which(exp.d1$stim.cat == 'light'),]


#obtainDataFromGroup(serie, baselineTrace = 200) 
#fwrite(temp, 'temp.csv') fwrite(temp, 'temp-thal.csv') fwrite(temp, 'temp-motor.csv')
temp<-fread('temp-thal.csv')
temp$sID<-as.character(temp$sID)
dataPeak(serie, 5000, 5060); peak.resp


viv.final<-viv.final[viv.final$Sample %in% sID.list]
#viv.final<-viv.final[!duplicated(viv.final$Sample),c('Sample','geno','Sex')]
viv.final<-rename(viv.final, c(Sample = 'sID'))
viv.final$sID<-as.character(viv.final$sID)

ddply(viv.final,
      .(geno, Sex),
      summarize,
      l=length(geno))

peak.resp<-merge(viv.final, peak.resp, by='sID')
peak.resp$desired.mWmm2<-as.numeric(peak.resp$desired.mWmm2)
peak.resp$depth<-factor(peak.resp$depth, levels = c(-300,-800))
peak.resp$geno<-factor(peak.resp$geno, levels = c('wt','het'))

peak.resp[minPeak == min(peak.resp$minPeak),]
peak.respLigth<-peak.resp


quartz(,2.8,1.6)
quartz(,2,2)
windows()
color<-c('blue','red')
toplottv<-c(0,1,5)
toplottv2<-c(0)


bpeakR<-peak.resp[, .(mean=mean(minPeak), lwr=mean(minPeak)-sd(minPeak)/sqrt(length(minPeak)), upr=mean(minPeak)+sd(minPeak)/sqrt(length(minPeak))), by=c('geno','depth','desired.mWmm2')]

bpeakR$depth<-factor(bpeakR$depth, levels = c(-300,-800))
bpeakR$geno<-factor(bpeakR$geno, levels = c('wt','het'))

ggplot(subset(bpeakR, depth=='-800' & !desired.mWmm2==60), aes(x=desired.mWmm2, y=mean, fill=geno, color=geno)) +
  
  geom_line(aes(color = geno), size=1, alpha=1)+
  #facet_wrap(~depth)+
  geom_ribbon(aes(ymax=upr,ymin=lwr, fill = geno), color=NA, alpha =.3)+
  geom_point(alpha=0.5)+
  
  
  
  #scale_y_continuous(limits=c(-1.5, 0.25),                           # Set y range
  #                   breaks=-6:20000 * 0.5) +                      # Set tick every 4
  #labs(title="Peak amplitude - supp layer (-300 um) & deep (-800 um) layers")+
  scale_fill_manual(values=color)+
  scale_color_manual(values=color)+
  xlab("Power (mW/mm2)")+
  ylab("LFP (mW)")+
  theme.tv+
  theme(legend.position = 'NONE')
quartz.save('fig-pilotdata_supp10.pdf', type = 'pdf')


fit<-aov(minPeak ~ geno*desired.mWmm2 + Error(sID), subset(peak.resp, !desired.mWmm2 %in% toplottv))
summary(fit)


ggplot(subset(peak.resp, !sID %in% toplottv2 & !desired.mWmm2 %in% toplottv), 
       aes(x=desired.mWmm2, y=minPeak, group=geno, color=geno)) +
  geom_smooth(method = "lm", aes(fill=geno), alpha=0.4)+
  facet_wrap(~depth)+
  geom_point(alpha=0.2)+
  geom_line(aes(group=sID, color = geno), size=1, alpha=0.2)+
  scale_y_continuous(limits=c(-1.5, 0.25),                           # Set y range
                     breaks=-6:20000 * 0.5) +                      # Set tick every 4
  labs(title="Peak amplitude - supp layer (-300 um) & deep (-800 um) layers")+
  scale_fill_manual(values=color)+
  scale_color_manual(values=color)+
  xlab("Power (mW/mm2)")+
  ylab("LFP (mW)")+
  theme.tv
quartz.save('pilotdata_supp2.pdf', type = 'pdf')
dev.size()


####### bar plot for peak amplitude #####

quartz(width=2.35,height=3.13)

peak.resp$depth<-factor(peak.resp$depth, levels = c(-300,-800))
peak.resp$geno<-factor(peak.resp$geno, levels = c('wt','het'))


give.n <- function(x){
        return(c(y=0.10, label = length(x))) 
        # experiment with the multiplier to find the perfect position
}
cbPalette<-c('#4758A6','#BC0404','#A8ABD5','#DD9E89')
ggplot(subset(peak.resp, !sID==112 & desired.mWmm2 == 10), aes(x=geno, y=-minPeak, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ depth)+
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
        ylab("LFP (-mV) at 10mW/mm2") +
        ggtitle("") +
        scale_y_continuous(limits=c(0, 1.5),                           # Set y range
                           breaks=0:1000 * 0.5,
                           expand = c(0,0)) +                      # Set tick every 4
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
quartz.save('pilotbarplot.pdf', type='pdf')

# CORRECTED
fit<-lm(minPeak~geno*depth,subset(peak.resp, desired.mWmm2==10 & !sID==112),
        contrasts=list(depth=contr.sum, geno=contr.sum))
Anova(fit, type=3)



to.rm()
t.test(minPeak ~ geno, var.equal=TRUE, 
       data=subset(peak.resp, desired.mWmm2==10 & depth == -800 & !sID==112))

######## post-hoc ########
summary(glht(fit, lsm(pairwise ~ geno | drug)))
summary(glht(fit, lsm(pairwise ~ drug | geno)))


####### EXTRA PILOT WHISKER ####

exp.d2<-exp.d1[which(exp.d1$stim.cat == 'piezo'),c('group','sID')]
tom1<-exp.d2[!duplicated(exp.d2$group),]
tom1<-merge(tom1, viv.final, by='sID')
serie<-na.omit(unique(exp.d2$group))

graphWisker(serie)

graphTraceMain(to.graph = temp,
               groupCluster = serie,
               scale=5,
               display.begin=000,
               display.end=510); graphTraceMainOutput


test<-peak.resp
test<-test[,c('sID','whisker','minPeak')]
test<-dcast(test, sID~whisker ,value.var = 'minPeak')
test[is.na(test)]<-0
test<-melt(test, id='sID')


test<-merge(viv.final, test, by='sID')
test[, `:=`(peak=min(value), sum=sum(value)), by=c('sID')]
test1<-test[!duplicated(test$sum),]
test1$geno<-factor(test1$geno, levels = c('wt','het'))

test2<-test[which(test$value==test$peak),]
test2<-test2[,c(1:4)]
test2[order(geno),]
colnames(test2)<-c('sID','geno','Sex','whisker')


test3<-merge(temp, test2, by=c('sID','whisker'))
str(b)
to.graph = b
groupCluster = 53
scale=50
display.begin=4990
display.end=5090

quartz(,6,3)
color<-c('blue','red')
b<-test3[, .(mean=mean(value), lwr=mean(value)-sd(value)/sqrt(length(value)), upr=mean(value)+sd(value)/sqrt(length(value))), by=c('timeMs','geno')]


ggplot(subset(b, timeMs<=1000 | timeMs>=1110 & timeMs>=0*10 & timeMs<=290*10), aes(x=timeMs, y=mean, color=geno, fill=geno))+
        #facet_wrap(~depth)+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
        #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        geom_ribbon(aes(ymax=upr,ymin=lwr, fill = geno), color=NA, alpha =.3)+
        geom_line(size=1)+
        scale_color_manual(values=color)+
        scale_fill_manual(values = color)+
        ylab("LFP (mV) at max whisker") +
        scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                           labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
        
        theme.tv+
        theme(legend.position = 'NONE')
quartz.save('whisk-trace2.pdf', type='pdf')

ggplot(subset(test3, timeMs<=1000 | timeMs>=1110 & timeMs>=0*10 & timeMs<=290*10), aes(y=value, x=timeMs, color=geno))+ #subset(temp, id>=0 & id<=4000)
        #geom_point()+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
        facet_wrap( ~depth)+
        geom_line(aes(group=sID), alpha = 0.6)+
        scale_color_manual(values=color)+
        ylab("LFP (mV) at max whisker") +
        scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                           labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
        theme.tv
        theme(legend.position = 'NONE')
quartz.save('whisk-trace1.pdf', type='pdf')


give.n <- function(x){
        return(c(y=010,, label = length(x))) 
        # experiment with the multiplier to find the perfect position
}
cbPalette<-c('#4758A6','#BC0404','#A8ABD5','#DD9E89')
ggplot(test1, aes(x=geno, y=-sum, color = geno, label=sID)) + 
        # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        #facet_wrap(~ depth)+
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
        ylab("Sum of LFP (-mV) peak resp. \n for 9 stimulated whiskers") +
        ggtitle("") +
        #scale_y_continuous(limits=c(0, 1.5),                           # Set y range
        #                   breaks=0:1000 * 0.5,
        #                   expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        geom_text(hjust = 0.05, nudge_x = 0.05, vjust = 0.00, nudge_y = 0.00, color='grey')+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = 8, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                axis.line.x = element_blank(),
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
quartz.save('whisksum.pdf',type='pdf')

t.test(test1$peak ~ test1$geno)

ggplot(test1, aes(x=geno, y=-peak, color = geno, label=sID)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        #facet_wrap(~ depth)+
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
        ylab("LFP (-mV) for the one whisker \n that had the max resp.") +
        ggtitle("") +
        #scale_y_continuous(limits=c(0, 1.5),                           # Set y range
        #                   breaks=0:1000 * 0.5,
        #                   expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        geom_text(hjust = 0.05, nudge_x = 0.05, vjust = 0.00, nudge_y = 0.00, color='grey')+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = 8, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                axis.line.x = element_blank(),
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
quartz.save('whiskpeak.pdf',type='pdf')

test$geno<-factor(test$geno, levels = c('wt','het'))
quartz(,3.6,1.8)
ggplot(test, aes(x=geno, y=-value, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ variable, ncol=9)+
        #geom_line(aes(fill=sID))+
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
        ylab("LFP (-mV) peak amplitude \n after whisker single stim") +
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
                strip.text.x = element_text(size = 8, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                axis.line.x = element_blank(),
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
quartz.save('whiskpeakALL.pdf',type='pdf')
getwd()


####### whisker test ####
test1<-test1[,sum.minuspeak:=sum-peak]

####### graph of trace with geno elaborated ####
temp$sID<-as.character(temp$sID)
temp<-merge(viv.final, temp, by='sID')

temp$depth<-factor(temp$depth, levels = c(-300,-800))
temp$geno<-factor(temp$geno, levels = c('wt','het'))




b<-temp[, .(mean=mean(value), lwr=mean(value)-sd(value)/sqrt(length(value)), upr=mean(value)+sd(value)/sqrt(length(value))), by=c('timeMs','geno','depth','desired.mWmm2')]
fwrite(b, 'tempgraphtest.csv')
b<-fread('tempgraphtest.csv')

b$depth<-factor(b$depth, levels = c(-300,-800))
b$geno<-factor(b$geno, levels = c('wt','het'))

str(b)
to.graph = b
unique(temp$desired.mWmm2)
groupCluster<-temp[,group][1]  #207 204
scale=500 
display.begin=4980
display.end=5340

quartz()
ggplot(subset(to.graph, depth=='-800' & timeMs>=display.begin*10 & timeMs<=display.end*10 & desired.mWmm2 == '10'), aes(x=timeMs, y=mean, color=geno, fill=geno))+
        facet_wrap(~depth)+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
        #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        geom_ribbon(aes(ymax=upr,ymin=lwr, fill = geno), color=NA, alpha =.3)+
        geom_line(size=1)+
        scale_color_manual(values=color)+
        scale_fill_manual(values = color)+
        ylab("Motor stim - LFP (mV) at 10mW/mm2") +
        scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                           labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
        
        theme.tv+
        theme(legend.position = 'NONE')
quartz.save('suppdata10.pdf', type='pdf')

quartz(,4,3)
ggplot(subset(to.graph, depth=='-800' & timeMs>=display.begin*10 & timeMs<=display.end*10 & desired.mWmm2 == 10), aes(x=timeMs, y=mean, color=geno, fill=geno))+
        #facet_wrap(~depth)+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =1, color="#27aae1")+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =1, color="#27aae1")+
        #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
        geom_hline(yintercept = 0, color = "black")+
        geom_ribbon(aes(ymax=upr,ymin=lwr, fill = geno), color=NA, alpha =.3)+
        geom_line(size=1)+
        scale_color_manual(values=color)+
        scale_fill_manual(values = color)+
        ylab("LFP (mV)") +
        labs(title='at 10mW/mm2')+
        xlab('Time (ms)')+
        scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                           labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)-5000))+
        #geom_segment(x=50000, xend=50020, y=0.2, yend=-0,1)+
        theme.tv+
        theme(axis.text.x = element_text(size=8))+
        theme(legend.position = 'NONE')

quartz.save('fig-suppdata10.pdf', type='pdf')

# CORRECTED
fit<-aov(value~geno*timeMs+Error(sID/timeMs),
         subset(temp, timeMs>=4090*10 & timeMs<=5060*10 & desired.mWmm2 == 10 & depth==-800)); summary(fit)
         
#contrasts=list(timeMs=contr.sum, geno=contr.sum)
#Anova(fit, type=3)

######## post-hoc ########
summary(glht(fit, lsm(pairwise ~ geno | drug)))
summary(glht(fit, lsm(pairwise ~ drug | geno)))


cbPalette<-c('#4758A6','#BC0404','#A8ABD5','#DD9E89')
graphTraceGenosingle <-function(to.graph = temp,
                     groupCluster = 13,
                     scale=c(5,50), 
                     display.begin=4990,
                     display.end=5090){
        
        #quartz(width = 6.6, height = 2.8)
        quartz(,6,2.8)
        graphTraceOutputGeno<<-ggplot(subset(temp, timeMs>=display.begin*10 & timeMs<=display.end*10 & desired.mWmm2==10), 
                                  aes(y=value, x=timeMs, group=sID, color=geno))+ #subset(temp, id>=0 & id<=4000)
                #geom_point()
                geom_line(alpha = 1)+
                theme.tv+
                facet_wrap( ~depth)+
                scale_color_manual(values=color)+
                geom_smooth(aes(geno))
                #labs(color="Laser power")+
                labs(title=paste("Motor cortex - " , "sID: ", temp[group==groupCluster, sID][[1]],
                                 " - stim for ", temp[group==groupCluster, stimDur][[1]],  " ms", sep=""),
                     subtitle=paste("recording S1",  temp[group==groupCluster, depth][[1]], "um", sep=""),
                     " - ", temp[group==groupCluster, date][[1]])+
                geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
                geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
                #geom_hline(yintercept = -0.2, linetype = 8, color = "grey")+
                geom_hline(yintercept = 0, color = "black")+
                xlab("Time (ms)")+
                scale_color_discrete(name = "power \n(mW/mm2)")+
                scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                                   labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
                #scale_y_continuous(limits=c(-0.3, 0.1),                           # Set y range
                #                   breaks=-10:1000 * 0.05,
                #                   expand = c(0,0))+
                ylab("LFP (mV)") +
                #scale_colour_manual(labels=c("10%", "50%", "100%", "565nm laser"), values=cbPalette)+
                theme.tv
}

quartz(,4,3)
color<-c('blue','red')
ggplot(subset(temp, timeMs>=4990*10 & timeMs<=5090*10 & desired.mWmm2==10), aes(y=value, x=timeMs, color=geno))+ #subset(temp, id>=0 & id<=4000)
        #geom_point()+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, linetype =8, color="red")+
        geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, linetype =8, color="red")+
        facet_wrap( ~depth)+
        geom_line(aes(group=sID), alpha = 0.6)+
        scale_color_manual(values=color)+
        ylab("LFP (mV) at 10mW/mm2") +
        scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                           labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))+
        theme.tv+
        theme(legend.position = 'NONE')
quartz.save('suppdata10-trace.pdf', type='pdf')

       

testab<-b[1:20,]
        

# whisker light comparison #####
test1<-test1[,c(1:3,6)]
colnames(test1)<-c('sID','geno','Sex','peakWhisk')

peakRL10<-peak.respLigth[desired.mWmm2==10,]
testfinal1<-merge(peakRL10, test1, by=('sID'))

quartz()
ggplot(subset(testfinal1, geno.y=='wt' & depth==-300), aes(x=-`peakWhisk`, y=-`minPeak`, color=geno.y))+ #[ which(data931$format == 384), ]
        geom_smooth(method="lm", aes(fill=geno.y))+
        geom_point()

totestcor<-subset(testfinal1, geno.y=='wt' & depth==-800)
cor(totestcor$`minPeak`, totestcor$`peakWhisk`); cor.test(totestcor$`minPeak`, totestcor$`peakWhisk`)

quartz()

testfinal1$depth<-factor(testfinal1$depth, levels = c(-300,-800))
ggplot(subset(testfinal1, desired.mWmm2 == 10), aes(x=-peakWhisk, y=-minPeak, color=geno))+ #[ which(data931$format == 384), ]
        geom_smooth(method="lm", aes(fill=geno))+
        geom_point(size=2)+
        facet_wrap(~ depth)+
        theme_bw()+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        ylab('light stim LFP(-mV) @ 10mW/mm2')+
        xlab('whisker stim LFP(-mV) max resp')+
        theme(axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              panel.background = element_blank())
        #theme(legend.position="NONE") 
quartz.save('suppdata101-trace.pdf', type='pdf')

testfinal1
testfinal2<-dcast(testfinal1, sID+desired.mWmm2+whisker+geno+Sex~depth, value.var = 'minPeak')

test<-dcast(peak.resp, sID+desired.mWmm2+geno+Sex~depth, value.var = 'minPeak')
test<-test[desired.mWmm2==10 & geno=='wt',]
fit<-aov(min ~ geno*desired.mWmm2, peak.resp)
cor(test$`-300`, test$`-800`); cor.test(test$`-300`, test$`-800`)

quartz()
ggplot(subset(test), aes(x=-`-300`, y=-`-800`, color=geno))+ #[ which(data931$format == 384), ]
        geom_smooth(method="lm", aes(fill=geno))+
        geom_point()


ggplot(subset(testfinal2, desired.mWmm2 == 10), aes(x=-`-300`, y=-`-800`, color=geno))+ #[ which(data931$format == 384), ]
        geom_smooth(method="lm", aes(fill=geno))+
        geom_point(size=2)+
        theme_bw()+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        ylab('light stim deep layer LFP(-mV) @ 10mW/mm2')+
        xlab('light stim supp layer LFP(-mV) @ 10mW/mm2')+
        theme(axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              panel.background = element_blank())
#theme(legend.position="NONE") 
quartz.save('suppdata102-trace.pdf', type='pdf')





gplot(subset(peak.resp, !sID %in% toplottv2 & !desired.mWmm2 %in% toplottv), 
      aes(x=desired.mWmm2, y=minPeak, group=geno, color=geno)) +
        geom_smooth(method = "lm", aes(fill=geno), alpha=0.4)+
        facet_wrap(~depth)+
        geom_point(alpha=0.2)+
        geom_line(aes(group=sID, color = geno), size=1, alpha=0.2)+
        scale_y_continuous(limits=c(-1.5, 0.25),                           # Set y range
                           breaks=-6:20000 * 0.5) +                      # Set tick every 4
        labs(title="Peak amplitude - supp layer (-300 um) & deep (-800 um) layers")+
        scale_fill_manual(values=color)+
        scale_color_manual(values=color)+
        xlab("Power (mW/mm2)")+
        ylab("LFP (mW)")+
        theme.tv



########## next pilot overview #
# analyze first the motor cortex
unique(exp.d[which(exp.d$experiment == 'F0020' &
            exp.d$stim.site == 'thalamus' & 
            exp.d$post.stim.ms == 5000), 'group'])

exp.d[which(exp.d$group == 208),]
