library(plyr)
library(ggplot2)
library(data.table)
dir<-'Y:/Vaissiere/00-ephys temp/';for(i in dir){
  setwd(i)
  setwd(i)
}

58.5-4.8
patchcableligth<-read.csv('patchcableligth.csv', header = TRUE, colClasses = c('POSIXct','character', 'character'))
str(patchcableligth)
current<-Sys.time()
ligthmeasure<-5.52
Dinolight<-9.58
patchcableligthTEMP<-data.table(current,ligthmeasure,Dinolight)
patchcableligthTEMP<-rbind(patchcableligth, patchcableligthTEMP)
write.csv(patchcableligthTEMP, 'patchcableligth.csv', row.names = FALSE)


# since 2/5/2018 measured baseline 
# 3.638 from the probe
toadd1<-read.csv('power.set1.csv', colClasses = c(rep("numeric",4),"POSIXct"))
#mat.lab.input<-c(0,0.1,1,2,3,4,5)
mat.lab.input<-c(0,0.1,0.5,1,2,3,4,5)
# old measured.value<-c(0,0.072,0.722,1.270,1.857,2.328,2.751) # the measured values are in mW
#measured.value<-c(0,0.081,1.128,2.12,2.98,3.74,4.38) # the measured values are in mW
measured.value<-c(0.073,0.0994,0.267,0.482,0.903,1.31,1.69,2.06) # the measured values are in m

# power analysis extract coef
#toadd1<-read.csv('power.set1.csv', colClasses = c(rep("numeric",4),"POSIXct"))
#mat.lab.input<-c(0,0.1,1,2,3,4,5)
#measured.value<-c(0,0.098,0.395,0.737,1.035,1.300,1.537) # the measured values are in mW

#last meas for 10mm
#mat.lab.input<-c(0,0.1,1,2,3,4,5)
#measured.value<-c(0,0.046,0.464,0.861,1.204,1.510,1.786) # the measured values are in mW


data.table(measured.value, measured.value-measured.value*val)

power.dat<-data.table(mat.lab.input,measured.value)

mod<-lm(measured.value ~ mat.lab.input, power.dat); summary(mod)

power.dat$intercept<-coef(mod)[1]
power.dat$slope<-coef(mod)[2]
power.dat$date.time<-Sys.time()


windows(); quartz()
ggplot(power.dat, aes(x=mat.lab.input, y=measured.value, group = 1)) +
  geom_smooth(se = TRUE, method = "lm", color="black")+
  geom_point()

power.dat<-rbind(power.dat, toadd1)

write.csv(power.dat, "power.set1.csv", row.names = FALSE)

# series for desried power
toadd2<-read.csv('power.set2.csv', colClasses = c(rep("numeric",3),"POSIXct"))
#pow.set<-data.table(desired.mWmm2=c(0.5,1,5,10,25,50,75))
pow.set<-data.table(desired.mWmm2=c(0,1,5,10,15,20,25,30,35,40))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2])
pow.set$date.time<-Sys.time()
ab<-print(pow.set[c(1,3)], digits = 1)
pow.set<-rbind(pow.set, toadd2)
write.csv(pow.set, "power.set2.csv", row.names = FALSE)


# matlab equivalent for following values
mat.equi<-data.table(mat.val=c(0.1,0.25,0.5,0.75,1,1.5,2,2.4,3,3.5,4,4.5,5))
mat.equi<-ddply(mat.equi,
                .(mat.val),
                transform,
                mW.mm2=(mat.val*coef(mod)[2])/(0.1*0.1*pi)); mat.equi
ab<-print(mat.equi[c(1,2)], digits = 2)

# new power set ####
pow.set<-data.table(desired.mWmm2=c(0,1.4,2.8,4.2,5.6,7,8.4,10,20,40))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2]); pow.set #-coef(mod)[1]
pow.set$date.time<-Sys.time()
print(pow.set[c(1,3)], digits = 2)
pow.set<-rbind(pow.set, toadd2)

# new power set ####
pow.set<-data.table(desired.mWmm2=c(0,2.5,5,7.5,10,12.5,15,17.5,20,30,40,50,60))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2]); pow.set #-coef(mod)[1]
pow.set$date.time<-Sys.time()
print(pow.set[c(1,3)], digits = 2)
pow.set<-rbind(pow.set, toadd2)
