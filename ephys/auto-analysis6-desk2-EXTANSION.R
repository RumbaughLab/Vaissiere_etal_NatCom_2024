temp<-fread('temp.csv')
temp[, valueP1:=c(NA,value[-.N]), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp[, delta:=valueP1-value, by=c('group','sID','depth','desired.mWmm2','whisker')]


#extract response 1
t1<-temp[temp[,.I[value<baseline3SD & timeMs>=stimOnset*10], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp1.time=timeMs[1], resp1.value=value[1]),by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)

#extract response 2
t1<-temp[temp[,.I[delta<=0 & timeMs>=resp1.time], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp2.time=timeMs[1]-1), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)              
t1<-temp[timeMs==resp2.time, .(resp2.value=value), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   

#extract response 4
t1<-temp[temp[,.I[timeMs>=stimOnset*10 & timeMs<=(stimOnset+200)*10], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp4.value=min(value)), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   
t1<-temp[value==resp4.value, .(resp4.time=timeMs), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   

#extract response 3
t1<-temp[temp[,.I[timeMs>=resp2.time & timeMs<=resp4.time], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp3.value=max(value)), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   
t1<-temp[value==resp3.value, .(resp3.time=timeMs), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   

#extract response 5
t1<-temp[temp[,.I[timeMs>=resp4.time & timeMs<=(stimOnset+60)*10], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp5.value=max(value)), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   
t1<-temp[value==resp5.value, .(resp5.time=timeMs), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   

#extract response 2.1
t1<-temp[temp[,.I[timeMs>=resp2.time & timeMs<=resp3.time], by=c('group','sID','depth','desired.mWmm2','whisker')]$V1]
t1<-t1[,.(resp2.1.value=min(value)), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   
t1<-temp[value==resp2.1.value, .(resp2.1.time=timeMs), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp<-merge(temp,t1,by=c('group','sID','depth','desired.mWmm2','whisker'), all=TRUE)   


testoutput<-temp[temp[,.I[1], by=c('sID','desired.mWmm2','depth')]$V1][!1]


ddply(testoutput,
      .(c(sID,desired.mWmm2,depth)),
        summarize,
      length(value))
#testoutput<-temp[!duplicated(c(temp$group, temp$sID)),];testoutput

# simple max slope

resp.sumry2<-testoutput[,`:=`(slope1=(resp2.value-resp1.value)/(resp2.time-resp1.time),
                               slope2=(resp2.1.value-resp2.value)/(resp2.1.time-resp2.time),
                               slope3=(resp3.value-resp2.1.value)/(resp3.time-resp2.1.time),
                               slope4=(resp4.value-resp3.value)/(resp4.time-resp3.time),
                               slope5=(resp5.value-resp4.value)/(resp5.time-resp4.time) )]

#fwrite(resp.sumry2, 'multicomponentWaveFrom.csv')
resp.sumry2<-fread('multicomponentWaveFrom.csv')
resp.sumry2$sID<-as.character(resp.sumry2$sID)
resp.sumry2<-merge(viv.final, resp.sumry2, by='sID')



data.table(colnames(resp.sumry2)); abc<-colnames(resp.sumry2)


resp.sumry2[,18:34] <- lapply(resp.sumry2[,18:34], function(x) as.numeric(as.character(x)))
md<-melt(resp.sumry2, id=abc[1:17])

str(resp.sumry2)




## graph #####
md$depth<-factor(md$depth, levels = c(-300,-800))
md$geno<-factor(md$geno, levels = c('wt','het'))
md$variable<-factor(md$variable, levels = c('resp1.value','resp2.value','resp3.value','resp4.value','resp5.value','resp1.time','resp2.time','resp3.time','resp4.time','resp5.time','slope1','slope2','slope3','slope4','slope5'))


give.n <- function(x){
        return(c(y=0.10, label = length(x))) 
        # experiment with the multiplier to find the perfect position
}
cbPalette<-c('#4758A6','#BC0404','#A8ABD5','#DD9E89')
quartz(,8.8,3.6)
tvtxtsize<-8
ggplot(subset(md, desired.mWmm2 %in% c(10) & variable %in% c('resp1.value','resp2.value','resp3.value','resp4.value','resp5.value')), aes(x=geno, y=value.1, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ depth*desired.mWmm2*variable, ncol= 16)+
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
                #position=position_jitter(w=0.1, h=0)
                ) + # size set the size of the point # alpha is setting the transparency of the point
        
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
        ylab("LFP (mV)") +
        ggtitle('resp.value for 10mW/mm2') +
        #scale_y_continuous(limits=c(0, 1.5),                           # Set y range
        #                   breaks=0:1000 * 0.5,
        #                   expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = tvtxtsize, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                #axis.line.x = element_blank(),
                axis.text = element_text(size=tvtxtsize, color = "black"),
                axis.title = element_text(size = tvtxtsize, color = "black"),
                axis.line.x = element_blank(),
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
quartz.save('multicomponentWaveFrom_V-P10.pdf', type='pdf')

quartz()
ggplot(subset(md, desired.mWmm2 %in% c(10) & variable %in% c('resp1.time','resp2.time','resp3.time','resp4.time','resp5.time')), aes(x=geno, y=(value.1-50000)/10, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ depth*desired.mWmm2*variable, ncol= 16)+
       
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
        ylab("Time (ms)") +
        ggtitle('resp.time for 10mW/mm2') +
        #scale_y_continuous(limits=c(0, 1.5),                           # Set y range
        #                   breaks=0:1000 * 0.5,
        #                   expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = tvtxtsize, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                axis.line.x = element_blank(),
                axis.text = element_text(size=tvtxtsize, color = "black"),
                axis.title = element_text(size = tvtxtsize, color = "black"),
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
quartz.save('multicomponentWaveFrom_T-P10.pdf', type='pdf')

md[variable=='resp4.time' & desired.mWmm2==5,]
devs<-dev.size()
unique(md$variable)
ggplot(subset(md, desired.mWmm2 %in% c(10) & variable %in% c('slope1','slope2','slope3','slope4','slope5')), aes(x=geno, y=value.1, color = geno)) + # 
        #geom_point(size=2, alpha=0.3, position = position_jitterdodge(jitter.width = 0.5,
        #                                                         jitter.height = 0,
        #                                                     dodge.width = 0.6)) +
        facet_wrap(~ depth*desired.mWmm2*variable, ncol= 10)+
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
        ylab("Slope")+
        #ylab("LFP (-mV) at 10mW/mm2") +
        ggtitle("slope for 10mW/mm2") +
        scale_y_continuous(limits=c(-0.05, 0.05),                           # Set y range
                           breaks=-100:1000 * 0.01,
                           expand = c(0,0)) +                      # Set tick every 4
        #scale_x_discrete(labels=c("c\n1","c\n2","c\n3","c\n4"))+
        scale_fill_manual(values=cbPalette)+
        #scale_colour_manual(values=cbPalette)+
        #theme_bw()+
        theme( #strip.text.x = element_blank(),
                #strip.background = element_blank(),
                strip.text.x = element_text(size = tvtxtsize, colour = "black", angle = 0),
                strip.background = element_rect(fill="grey85", colour="black"),
                axis.line = element_line(color="black", size=0.5),
                axis.line.x = element_blank(),
                axis.text = element_text(size=tvtxtsize, color = "black"),
                axis.title = element_text(size = tvtxtsize, color = "black"),
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
quartz.save('multicomponentWaveFrom_S-P10.pdf', type='pdf')

############### JUNK BELOW ########################


merge(t1,t2,by=c('group','sID','depth','desired.mWmm2','whisker'))

`:=`(resp4.value=min(value)),


DT1<-DT[DT[,.I[B>=30 & B<=40], by='A']$V1]
DT1[, max(C), by='A']



#extract respon 3
temp[timeMs>=resp2.time & timeMs<=resp4.time]
temp[value==resp3.value, `:=`(resp3.time=timeMs), by=c('group','sID','depth','desired.mWmm2','whisker')]

temp[delta<=0 & timeMs>=resp2.time,`:=`(resp2.1.time=timeMs[1]-1), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp[,`:=`(resp2.1.time=shift(resp2.time, 1L, type='lead'))]
temp[timeMs==resp2.1.time, resp2.1.value:=value, by=c('group','sID','depth','desired.mWmm2','whisker')]

temp[delta<=0 & timeMs>=resp2.1.time,`:=`(resp3.time=timeMs[1]-1), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp[,`:=`(resp3.time=shift(resp3.time, 1L, type='lead'))]
temp[timeMs==resp3.time, resp3.value:=value, by=c('group','sID','depth','desired.mWmm2','whisker')]

temp[delta<=0 & timeMs>=resp2.1.time,`:=`(resp3.time=timeMs[1]-1), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp[,`:=`(resp3.time=shift(resp3.time, 1L, type='lead'))]
temp[timeMs==resp3.time, resp3.value:=value, by=c('group','sID','depth','desired.mWmm2','whisker')]


# simplified variables to obtain script above
#resp1<-y1[value<baseline3SD & timeMs>=f.list[i, "stim.onset"]*10,timeMs][1]
#resp2<-y1[delta<=0 & timeMs>=comp1.intial,timeMs][1] #comp1 correspond to the interval of the first component
#resp4<-y1[value==min(value),timeMs-1][1]
#resp3<-y1[timeMs>=resp2 & timeMs<=resp4,][value==max(value),timeMs]
#resp5<-y1[value==max(value) & timeMs>=resp4, timeMs][1]
#resp2.1<-y1[timeMs>=resp2 & timeMs<=resp3,][value==min(value),timeMs]




temp[3458,]

DT <- data.table(A=1:5, B=1:5*10, C=1:5*100)
DT[ , D := C + shift(B, 1L, type="lag")]
DT[ , D := shift(B, 1L, type='lead')[1]]

temp[timeMs==resp2.time,.(resp2.value=value), by=c('group','sID','depth','desired.mWmm2','whisker')]
temp[value==resp4.value,]
temp[timeMs==resp2.time,]
unique(temp$resp2.value)





test[4934:4936,]

temp<-temp1
temp<-temp1[group==201]





# simple max slope

resp.sumry2<-resp.sumry1[,`:=`(slope1=(resp2.value-resp1.value)/(resp2.time-resp1.time),
                               slope2=(resp2.1.value-resp2.value)/(resp2.1.time-resp2.time),
                               slope3=(resp3.value-resp2.1.value)/(resp3.time-resp2.1.time),
                               slope4=(resp4.value-resp3.value)/(resp4.time-resp3.time),
                               slope5=(resp5.value-resp4.value)/(resp5.time-resp4.time) )]





groupCluster=200
scale=500
quartz()
        ggplot(subset(temp, timeMs>=1300 & timeMs<=1400 ), 
                          aes(y=value, x=timeMs))+ #subset(temp, id>=0 & id<=4000)
        geom_point(alpha = 1)+

                #geom_vline(xintercept = 2000)+
                geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10, color="#1dcaff")+
                geom_vline(xintercept = temp[group==groupCluster, stimOnset][[1]]*10+temp[group==groupCluster, stimDur][[1]]*10, color="#1dcaff")+
                
                geom_vline(xintercept = temp$resp1.time, color="red")+
                geom_vline(xintercept = temp$resp2.time, color="red")+
                geom_vline(xintercept = temp$resp3.time, color="red")+
                geom_vline(xintercept = temp$resp4.time, color="red")+
                geom_vline(xintercept = temp$resp5.time, color="red")+
                geom_vline(xintercept = temp$resp2.1.time, color="blue")+
        facet_wrap(~group)
                
                
                
                geom_vline(xintercept = y1[value<baseline3SD & timeMs>=1000,timeMs][1], linetype =8, color="red")+
                geom_vline(xintercept = y1[value==min(value),][1], linetype =8, color="red")+
                geom_vline(xintercept = y1[delta<=0 & timeMs>=y1[value<baseline3SD & timeMs>=1000,timeMs][1],timeMs][1], linetype =8, color="red")+
                scale_x_continuous(breaks=seq(0,temp[group==groupCluster, recDur][[1]]*10, by = temp[group==groupCluster, recDur][[1]]*10/scale),
                                   labels=as.character(seq(0,temp[group==groupCluster, recDur][[1]], by = temp[group==groupCluster, recDur][[1]]/scale)))


a<-(y1[(f.list[i, "stim.onset"]*10-baselineTrace*10):(f.list[i,"stim.onset"]*10), value])


dataPeak<-function(groupCluster = 18,
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



########## EXTENDED ANALYS   ##
y1[which(value<baseline3SD),]

# value for the first part of the slope going down
y1[value<baseline3SD & timeMs>=f.list[i, "stim.onset"]*10,timeMs][1]

# find the first peak
y1[, valueP1:=c(NA,value[-.N])]
y1[, delta:=valueP1-value]

output<-y1[timeMs>=lowerInterval*10 & timeMs<=upperInterval*10 & group == groupCluster,
           .(resp1)]


(-0.1093697--0.01021907)/(1034-1017)
resp.sumry2<-NULL
# slope formula for 80/20 slope
#(y1[timeMs==(b-as.integer((b-a)*0.2)), value]-y1[timeMs==(a+as.integer((b-a)*0.2)), value])/((b-as.integer((b-a)*0.2))-(a+as.integer((b-a)*0.2)))




resp.sumry1<-temp[,#.(resp1.time=temp[value<baseline3SD & timeMs>=stim.onset*10,timeMs][1],
                  #resp1.value=temp[value<baseline3SD & timeMs>=stim.onset*10,value][1],
                  #resp2.time=temp[delta<=0 & timeMs>=comp1.intial,timeMs-1][1],
                  #resp2.value=temp[temp[delta<=0 & timeMs>=comp1.intial,timeMs-1][1],value][1],
                  
                  resp2.1.time=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs-1][1] & timeMs<=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs][1] & timeMs<=temp[value==min(value),timeMs-1][1],][value==max(value),timeMs],][value==min(value),timeMs],
                  resp2.1.value=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs-1][1] & timeMs<=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs][1] & timeMs<=temp[value==min(value),timeMs-1][1],][value==max(value),timeMs],][value==min(value),value],
                  
                  resp3.time=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs][1] & timeMs<=temp[value==min(value),timeMs-1][1],][value==max(value),timeMs],
                  resp3.value=temp[timeMs>=temp[delta<=0 & timeMs>=comp1.intial,timeMs][1] & timeMs<=temp[value==min(value),timeMs-1][1],][value==max(value),value],
                  resp4.time=temp[value==min(value),timeMs-1][1],
                  resp4.value=temp[temp[value==min(value),timeMs-1],value][1],
                  resp5.time=temp[value==max(value) & timeMs>=temp[value==min(value),timeMs-1][1], timeMs][1],
                  resp5.value=temp[temp[value==max(value) & timeMs>=temp[value==min(value),timeMs-1][1], timeMs][1], value][1]),
by=c('group','sID','depth','desired.mWmm2','whisker')]; resp.sumry1
