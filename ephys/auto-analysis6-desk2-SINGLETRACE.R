##### exp.d
ab<-exp.d[which(exp.d$experiment=='F0010' & exp.d$desired.mWmm2==40 & exp.d$surface-exp.d$loc==-800), c('files','sID','desired.mWmm2')]
ab<-merge(ab, viv.final)
length(ab$files)
str(exp.d)
i<-"MATLA/2018/Mar/26/experiment001trial046.mat"
i<-1
singletrace<<-NULL
for(i in 1:nrow(ab)){
        data<-readMat(ab[i,'files']) 
        y1<-data.table(data$inputData)
        y1<-data.table(y1, avg.trace=apply(y1, 1, mean))
        y1$time<-as.numeric(row.names(y1))
        y1<-y1[48000:50600,]
        
        y1<-melt(y1, id="time")
        y1.m<-y1[time<50000, .(mean=mean(value)), by='variable']
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


singletrace$variable

graphSingleTrace<-function(to.graph = singletrace, subject = 111){
quartz()
        print(ggplot(subset(singletrace, time>=4990*10 & time<=5060*10 & sID==subject), aes(x=time-50000, y=norm, group = variable, alpha = prop.line, size = prop.line))+ 
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
        scale_y_continuous(limits=c(-2 , 1),                           # Set y range
                breaks=-10:1000 * 0.2)+
        scale_x_continuous(breaks=seq(0,10000*10, by = 10000*10/scale),
                                   labels=as.character(seq(0,10000, by = 10000/scale)))+
        theme.tv+
        theme(legend.position = 'NONE'))
+
        quartz.save(paste("OUTPUT DATA/",
                  "SINGLE-TRACE_", Sys.Date(),
                  "_sID_", singletrace[sID==subject, sID][1], 
                  "_geno_", singletrace[sID==subject, geno][1],
                  "_pwr_", singletrace[sID==subject, desired.mWmm2][1],
                  '.pdf',
                  sep=''),
            type='pdf') 
dev.off()
}

for(i in ab[,1]){print(i)}
for(i in ab[,1]){graphSingleTrace(singletrace, i)}
getwd
