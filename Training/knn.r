library(class)

separate <- function(df){
  # Set the seed for the test-training split
  set.seed(1);
  selector <- sample(c('Train','Test'), size=nrow(df), replace=T, prob=c(0.75,0.25))
  df["Selector"] <- selector;
  frames <- split(df,df$Selector);
  frames[[1]]$Selector <- NULL;
  frames[[2]]$Selector <- NULL;
  return(frames);
}

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

df1 <- read.csv("../Data/train.csv",header=FALSE)
df2 <- read.csv("../Data/test.csv",header=FALSE)
trainData <- data.frame(df1);
testData <- data.frame(df2);
  
trainData$V1 <- NULL;
testData$V1 <- NULL;

trainCats <- trainData$V273
trainData$V273 <- NULL;

trainData <- data.matrix(trainData);
testData <- data.matrix(testData);

dim(trainData)
dim(testData)
length(trainCats)

doKnn <- function(){
  ks <- c(1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,125,250,500,1000);
  trainerrs <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
  testerrs <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
  count <- 1;

  for (kval in ks){
    print(kval)
    print("Test")
    test <- knn(trainData,testData,cl=trainCats,k=kval)
    write.table(as.data.frame(test),paste('../Output/knn',kval,'.csv',collapse=""))
    print("Train")
    train <- knn(trainData,trainData,cl=trainCats,k=kval);
    traintab <- table(train,trainCats);
    trainerr <- accuracy(traintab);
    trainerrs[count] <- 100-trainerr;
    count <- count + 1;
  }

  png("../Plots/errorvsk.png");
  plot(ks,trainerrs,type="l",col="blue");
  dev.off();
  return(testerrs);
}

#partD <- function(){
#  dfCopy <- df;
#  # V2 is Clump Thickness
#  dfCopy$V2 <- dfCopy$V2 * 10;
#  ks <- c(1,2,3,4,5,6,7,8,9,10,20,30,40,50);
#  trainerrs <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0);
#  testerrs <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0);
#  count <- 1;
#
#  for (kval in ks){
#    test <- knn(trainData,testData,cl=trainCats,k=kval)
#    testtab <- table(test,testCats);
#    testerr <- accuracy(testtab);
#    testerrs[count] <- 100-testerr;
#    train <- knn(trainData,trainData,cl=trainCats,k=kval);
#    traintab <- table(train,trainCats);
#    trainerr <- accuracy(traintab);
#    trainerrs[count] <- 100-trainerr;
#    count <- count + 1;
#  }
#
#  png("errorvskPartD.png");
#  plot(ks,testerrs,type="l",col="blue");
#  lines(ks,trainerrs,col="red");
#  legend("topleft",c("Test error","Training Error"),fill=c("blue","red"))
#  dev.off();
#  return(testerrs);
#}

testc <- doKnn()

ks <- c(1,2,3,4,5,6,7,8,9,10,20,30,40,50);

#png("partCvsD.png")
#plot(ks,testc,type="l",col="blue");
#lines(ks,testd,col="red");
#legend("topleft",c("Part C","Part D"),fill=c("blue","red"))
#dev.off()
