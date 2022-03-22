#CIS code:tfgr67
install.packages("skimr")
install.packages("ggplot2")
install.packages("rsample")
install.packages("DataExplorer")
install.packages("boot")
install.packages("mlr3verse")
install.packages("dplyr")
install.packages("data.table")
install.packages("mlr")
install.packages("mlr3")
install.packages("caret")
library("skimr")
library("ggplot2")
library("rsample")
library("DataExplorer")
library("boot")
library("mlr3verse")
library("dplyr")
library("data.table")
library("mlr")
library("mlr3")
library("caret")
Heart_Failure<- read.csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv", header=TRUE)
skim(Heart_Failure)
DataExplorer::plot_histogram(Heart_Failure, ncol = 3)
DataExplorer::plot_boxplot(Heart_Failure, by = "fatal_mi", ncol = 3)
set.seed(212) # set seed for reproducibility

Heart_Failure$fatal_mi <- as.factor(Heart_Failure$fatal_mi)

fmi_task <- TaskClassif$new(id = "Heart",backend = Heart_Failure,target = "fatal_mi")
fmi_task_train = sample(fmi_task$row_ids, 0.7 * fmi_task$nrow)
fmi_task_test = setdiff(fmi_task$row_ids, fmi_task_train)

mlr_tasks$add("Heart", fmi_task)
tasks = lapply(c("Heart"), tsk)
learners = lrns(c("classif.log_reg", "classif.rpart", "classif.ranger"))
resamplings = list(rsmp("cv"), rsmp("bootstrap"))
grid = benchmark_grid(tasks, learners, resamplings)
fmi <- benchmark(grid, store_models = TRUE)

measures <- msrs(c("classif.ce","classif.prauc",'classif.auc'))
mm <- msrs(c("classif.precision","classif.recall",'classif.fbeta'))
mmm <- msrs(c("classif.tn","classif.fn",'classif.fp',"classif.tp"))
print(fmi$aggregate(measures))
print(fmi$aggregate(mm))
print(fmi$aggregate(mmm))

head(fortify(fmi))
autoplot(fmi) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 45,hjust = 1))
autoplot(fmi$clone(deep = T)$filter(task_ids = "Heart",resampling_ids = "cv"),type = "roc")
autoplot(fmi$clone(deep = T)$filter(task_ids = "Heart",resampling_ids = "bootstrap"),type = "roc")
autoplot(fmi$clone(deep = T)$filter(task_ids = "Heart",resampling_ids ="bootstrap"),type = "prc")
autoplot(fmi$clone(deep = T)$filter(task_ids = "Heart",resampling_ids ="cv"),type = "prc")
#Random Forest
lrn_RF <- lrn("classif.ranger", predict_type = "prob",importance = "permutation")
bootstrap <- rsmp("bootstrap")

set.seed(212)
Rtrain <- lrn_RF$train(fmi_task,row_ids = fmi_task_train)
Rtest = Rtrain$predict(fmi_task, row_ids = fmi_task_test)
Rtest$score(msr("classif.ce"))

importance = as.data.table(Rtrain$importance(), keep.rownames = TRUE)

colnames(importance) = c("name", "Importance")

ggplot(data=importance,aes(x = reorder(name, Importance), y = Importance)) + 
  geom_col(fill="blue",width=0.8) + coord_flip() + xlab("Variables'name")

Rtest = Rtrain$predict(fmi_task, row_ids = fmi_task_test)

set.seed(212)
Rtest$score(msrs(c("classif.ce","classif.prauc",'classif.auc')))
Rtest$confusion

res_RF <- resample(fmi_task, lrn_RF, bootstrap, store_models = TRUE)

set.seed(212)
res_RF$aggregate(list(msr("classif.ce"),msr("classif.prauc"),msr("classif.auc"),msr("classif.fpr"),msr("classif.fnr")))

lrn_RF$param_set

search_space <- ps(
  min.node.size = p_int(lower = 50, upper = 80),
  sample.fraction= p_dbl(lower = 0.65, upper = 0.71)
)
search_space

resampling <- rsmp("bootstrap")
measure <- msr("classif.ce")

auto_learner <- auto_tuner(
  learner = lrn_RF,
  resampling = resampling,
  measure = measure,
  search_space = search_space,
  method = "grid_search",
  term_evals = 20
)

Rtrain_ <- auto_learner$train(fmi_task)
Rtrain_$tuning_result

Rtest_ = Rtrain_$predict(fmi_task, row_ids = fmi_task_test)
Rtest_$score(msrs(c("classif.ce","classif.prauc",'classif.auc',)))
Rtest$score(msrs(c("classif.ce","classif.prauc",'classif.auc')))
Rtest_$confusion
Rtest$confusion

Heart.task = makeClassifTask(id = "HeartF", data = Heart_Failure, target = "fatal_mi")

Cali_Curve = list(makeLearner("classif.rpart", predict.type = "prob"))
trains = lapply(Cali_Curve, train, task = Heart.task)
predicts = lapply(trains, predict, task = Heart.task)
outcome = generateCalibrationData(predicts)
plotCalibration(outcome,rag = TRUE,reference = TRUE)






