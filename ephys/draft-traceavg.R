set.seed(1234)
a1<-rnorm(20, mean=10, sd=5)
a2<-rnorm(20, mean=50, sd=5)
a3<-rnorm(20, mean=310, sd=5)
a4<-rnorm(20, mean=450, sd=5)
a5<-seq(1,20,1)

a<-data.table(a1,a2,a3,a4,a5)
a<-melt(a, id='a5')

geno<-rep(c('het','wt'),each=40)
a<-cbind(a, geno)

b<-a[, .(value=mean(value)), by=c('a5','geno')]
b$variable<-'mean'


a<-rbind(a,b)

a$prop.line[a$variable == "mean"]<-1
a$prop.line[!a$variable == "mean"]<-0.1
a$variable<-paste(a$geno, a$variable)

ggplot(a, aes(x=a5, y=value, group=variable, color=geno, alpha = prop.line, size = prop.line))+
        scale_size(range = c(1, 6))+
        scale_alpha(range = c(0.5,1))+
        geom_line()+
        theme.tv

