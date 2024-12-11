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