### Photograph examples


## Read in the library and metadata
#install.packages("jpeg")
library(jpeg)

# pm <- read.csv("C:\\Users\\user\\Downloads\\photoMetaData.csv")
pm <- read.csv("photoMetaData.csv")
n <- nrow(pm) ##find number of rows in pm

trainFlag <- (runif(n) > 0.5) ##create random numbers (from zero to 1), create n random numbers, one for each row of data
                              ## and each random number will be tested against the boolean condition greater than 0.5 so that
                              ## we can later use this to randomly select train and test data

##trainFlag
##k <- runif(10)
##k

y <- as.numeric(pm$category == "outdoor-day") ##converts category to 1 if true and 0 is false
##y

##g <- as.numeric(1 == 1)
##g

X <- matrix(NA, ncol=3, nrow=n) ## create a matrix. matrix(values, number of colums, number of rows)
                                ## NA means that we create an empty matrix, which we will fill in later
                                ## by using a for loop and looping over all rows, j, in data of images
#C:\Users\Faizan\OneDrive\OneDrive\Statistical Cnslt and Tutoring\Kwadwo Crypto Currency\FinalProject_Image_Classification\columbiaImages\columbiaImages
for (j in 1:n) {
  img <- readJPEG(paste0("C:\\Users\\Faizan\\OneDrive\\OneDrive\\Statistical Cnslt and Tutoring\\Kwadwo Crypto Currency\\FinalProject_Image_Classification\\columbiaImages\\columbiaImages\\",pm$name[j])) 
                  ## in this line above paste0 says that we want to apply the file path name at the front
                  ## of whatever data is in the vector/column pm$name[j], so we iterate over the images located
                  ## in dataframe pm (which are the images we are working with) and the column of pm named "name1", name 2,name3...
                  ## it will iterate as per the for loop, which goes from 1 to n, so all rows and we will eventually
                  ## add this to the empty (n by 3 matrix) matrix we created just before the for loop
  X[j,] <- apply(img,3,median) ## apply function returns a vector, array, or list after applying something to img object, to either
                              ## 1, the rows, 2, the columns, or 3 (here our matrix is actually a 3 dimentional matrix, so margin is not just
                              ## related to either rows or columns, but also the third dimention (red, green, blue, opacity)) after applying the function (median in this case) to 
                              ##each img data point in vector/object img. Essentially, we are decreasing the dimentionality of the image data
                              ## turning a 3 dimentional matrix into a two dimentional matrix by using the median opacity 
  print(sprintf("%03d / %03d", j, n)) ##sprint tells R how to format specific data, j and n. This code means format j and n
                                      ## such that there are no decimals, and at least 3 digits in the output with numbers that are less
                                      ## than 3 digits having leading zeroes so the end up as 3 digits. it is printing out j/total number of rows (n)
                                      ## which is essentially showing the progress of the for loop 1/800, 2/800 ...etc
}

X



# build a glm model on these median values
out <- glm(y ~ X, family=binomial, subset=trainFlag)
out$iter
summary(out)

# How well did we do?
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])

## ROC curve (see lecture 12)
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
glmAuc <- auc(r)
glmAuc

##using EBImage, richer documentation
install.packages("BiocManager")
BiocManager::install("EBImage")
library("EBImage")


library(cluster)
featureCreation <- read.csv("C:\\Users\\user\\Downloads\\photoMetaData.csv")

image1Test <- readImage(paste0("C:\\Users\\user\\Downloads\\columbiaImages\\columbiaImages\\",pm$name[1]))
print(image1Test)
display(image1Test)

nOfRows <- nrow(featureCreation) ##find number of rows in pm
secondLevelofClustering <- as.numeric(pm$category == "outdoor-day") ##converts category to 1 if true and 0 is false
##y

## perform clustering metric is euclidian (could choose manhatten distance since no negative colors, but leave for now)
resultsOfCluster = agnes(featureCreation, diss=FALSE, metric="euclidian")
plot(resultsOfCluster)  ##dendogram

##now use the classes from the dendogram to create classifications
library(MASS)
n=nrow(pm) ##corrected -total images is 800
nt=640 ##80% of data - corrected for image data
neval=n-nt
rep=5
errlin=dim(rep)
errqua=dim(rep)
for (k in 1:rep) {
  train=sample(1:n,nt)
  ## linear discriminant analysis
  m1=lda(V21~.,germandata[train,],prior=c(.506,.494))
  predict(m1,germandata[-train,])$class
  tablin=table(germandata$V21[-train],predict(m1,germandata[-train,])$class)
  errlin[k]=(neval-sum(diag(tablin)))/neval
}
