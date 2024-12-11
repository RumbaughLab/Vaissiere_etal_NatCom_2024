library(data.table)
library(ggplot2)
library(tidyr)
library(plyr)
library(dplyr)
library(R.matlab)
# FUNCTION - THEME ####
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

graphSingleTrace<-function(to.graph = singletrace, subject = 538){
        quartz(, 3.5, 4.2)
        print(ggplot(subset(singletrace, time>=1800 & time<=3400 & sID==subject), aes(x=time-2000, y=norm, group = variable, alpha = prop.line, size = prop.line))+ 
                      geom_line(aes(color= geno=='wt'))+
                      scale_color_manual(values=setNames(c('blue','red'), c(T,F)))+
                      scale_size(range = c(0.4, 0.6))+
                      #scale_alpha(range = c(0.9, 1))+
                      labs(title=paste('-800 um @ ', singletrace[sID==subject, desired.mWmm2][1]), subtitle=paste('sID: ', singletrace[sID==subject, sID][1] , ' geno: ', singletrace[sID==subject, geno][1] ))+
                      xlab('Time (ms)')+
                      ylab('LFP (mV)')+
                      geom_vline(xintercept = 0*10, color="#1dcaff")+
                      geom_vline(xintercept =2*10, color="#1dcaff")+
                      scale_alpha(range = c(0.1,1))+
                      scale_y_continuous(limits=c(-1.4 , 1),                           # Set y range
                                         breaks=-10:1000 * 0.2)+
                      scale_x_continuous(breaks=seq(0,10000*10, by = 10000*10/scale),
                                         labels=as.character(seq(0,10000, by = 10000/scale)))+
                      theme.tv+
                      theme(legend.position = 'NONE')) +
                quartz.save(paste("OUTPUT DATA/F0021/",
                                  "SINGLE-TRACE_5", Sys.Date(),
                                  "_sID_", singletrace[sID==subject, sID][1], 
                                  "_geno_", singletrace[sID==subject, geno][1],
                                  "_pwr_", singletrace[sID==subject, desired.mWmm2][1],
                                  '.pdf',
                                  sep=''),
                            type='pdf') 
        #dev.off()
}

dataPeak<-function(lowerInterval = 211,
                   upperInterval = 280){
        peak.resp<<-singletrace[time>=lowerInterval*10 & time<=upperInterval*10 & sID %in% c(539,546) & !variable == 'avg.trace', 
                         .(minPeak=min(norm), 
                           maxPeak=max(norm),
                           minPeakTime=(lowerInterval*10+which.min(norm))/10,
                           maxPeakTime=(lowerInterval*10+which.max(norm))/10,
                           deltaTime=((lowerInterval*10+which.max(norm))/10-(lowerInterval*10+which.min(norm))/10),
                           slopeMinMax=(max(norm)-min(norm))/((lowerInterval*10+which.max(norm))/10-(lowerInterval*10+which.min(norm))/10)), 
                         by=c('variable','sID')]
}
##### exp.d ####
mainFolder = '/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/'
mainFolder = 'Y:/Vaissiere/ARCHIVE/00-ephys temp/'
dir<-setwd(mainFolder)
dir<-setwd(mainFolder)
# exp.d<-read.csv('/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/Description.csv')
exp.d<-read.csv('Description.csv')
exp.d$files<-paste("MATLA/",
                   exp.d$year,"/",
                   exp.d$month,"/",
                   formatC(exp.d$day, width = 2, flag="0"),"/",
                   "experiment", formatC(exp.d$exp, width = 3, flag="0"), "trial",formatC(exp.d$trial, width = 3, flag="0"),".mat",
                   sep = "")

ab<-exp.d[which(exp.d$experiment=='F0021'), c('files','sID','desired.mWmm2')]

viv.final<-data.frame(cbind(as.character(exp.d[which(exp.d$experiment=='F0021'), c('sID')]), rep('wt', 13)))
colnames(viv.final)<-c('sID','geno')
ab<-merge(ab, viv.final)


singletrace<<-NULL
for(i in 1:nrow(ab)){
        data<-readMat(ab[i,'files']) 
        y1<-data.table(data$inputData)
        y1<-data.table(y1, avg.trace=apply(y1, 1, mean))
        y1$time<-as.numeric(row.names(y1))
        y1<-y1[0:10000,]
        
        y1<-melt(y1, id="time")
        y1.m<-y1[time<2000, .(mean=mean(value)), by='variable']
        output<-merge(y1, y1.m, by="variable")
        output<-transform(output, 
                          norm = value - mean)
        output$prop.line[output$variable == "avg.trace"]<-1
        output$prop.line[!output$variable == "avg.trace"]<-0.8
        output$sID<-ab[i,'sID']
        output$geno<-ab[i,'geno']
        output$desired.mWmm2<-ab[i,'desired.mWmm2']
        singletrace[[i]]<-output
        print(paste('progress: ', i, '/', nrow(ab)))
}
singletrace<-rbindlist(singletrace)
singletrace<<-singletrace
scale<-500

first5<-c('V1','V2','V3','V4','V5')
last5<-c('V30','V29','V28','V27','V26')
str(singletrace)


quartz()
graphSingleTrace(subject=547)




dataPeak(220,240); peak.resp$wf<-'wf2' #window of analysis for the second peak 220/240 wf2
peak.resp2<-peak.resp
dataPeak(200,220); peak.resp$wf<-'wf1' #window of analysis for the second peak 220/240 wf2
peak.resp1<-peak.resp

peak.resp<-rbind(peak.resp1,peak.resp2)


grpahifor(i in ab[,1]){print(i)}
for(i in ab[,1]){graphSingleTrace(singletrace, i)}
getwd()

### for F0021a ####
dataPeak(dataLi, 5000, 5020); peak.resp
peak.resp1<-peak.resp[sID %in% c(539,546),]
ggplot(peak.resp, aes(x=sID, y=-minPeak, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ wf)+
        geom_bar(aes(fill=sID),
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
        
        #stat_summary(fun.data = give.n, color="white",geom="text",size=4)+
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
        ylab("Peak amplitude (mV)") +
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
quartz.save('F0021a.pdf', type='pdf')
