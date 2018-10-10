using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace MLParalel
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "SalaryData.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "SalaryData-test.csv");
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //  var pipe = CreateModel();


            ObservableCollection<ILearningPipelineItem> trainers = new ObservableCollection<ILearningPipelineItem>
            {
               new GeneralizedAdditiveModelRegressor(),
               new PoissonRegressor()
            };
            SetTrainers(trainers);

            Console.ReadLine();
        }


        static LearningPipeline CreateModel()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<SalaryData>(useHeader:true,separator:';'),
                new ColumnConcatenator("Features","YearsExperience")
            };

            //  var batata = new Microsoft.ML.Trainers.FastForestRegressor();


            return pipeline;

        }


        static void SetTrainers(ObservableCollection<ILearningPipelineItem> trainers)
        {
            var trainersCount = trainers.Count;


            var pipelines = new ObservableCollection<LearningPipeline>();

            for (int i = 0; i < trainersCount; i++)
            {
                var pipeline = new LearningPipeline
                {
                    new TextLoader(_datapath).CreateFrom<SalaryData>(useHeader:true,separator:';'),
                    new ColumnConcatenator("Features","YearsExperience")
                };
                pipeline.Add(trainers[i]);

                pipelines.Add(pipeline);
            }

            var evaluator = new RegressionEvaluator();
            var testeData = new TextLoader(_testdatapath).
                                CreateFrom<SalaryData>(true, ';');

            var watch = new Stopwatch();
            watch.Start();

            //foreach (var item in pipelines)
            //{
            //    var model = item.Train<SalaryData, SalaryPrediction>();

            //    var metrics = evaluator.Evaluate(model, testeData);

            //    Console.WriteLine($"RMS - {metrics.Rms}       - Test {pipelines.IndexOf(item)}");
            //    Console.WriteLine($"R²  - {metrics.RSquared}  - Test {pipelines.IndexOf(item)} ");
            //}


            //if i want to use tasks
            var tasks = new List<Task>();
            foreach (var item in pipelines)
            {
                var t1 =  Task.Factory.StartNew(() =>
                {
                    var model = item.Train<SalaryData, SalaryPrediction>();

                    var metrics = evaluator.Evaluate(model, testeData);


                    Console.WriteLine($"RMS - {metrics.Rms}       - Test {pipelines.IndexOf(item)}");
                    Console.WriteLine($"R²  - {metrics.RSquared}  - Test {pipelines.IndexOf(item)} ");
                });

                tasks.Add(t1);
            }
            //if Using new Task( ()=> { } );
            // tasks.StartAll();

            Task.WaitAll(tasks.ToArray());

            watch.Stop();
            Console.WriteLine();
            Console.WriteLine($"Time elapsed : {watch.ElapsedMilliseconds}ms");


            //My best times are: 
            //5201ms->in Main thread
            //2508ms->using new Task( ()=> { } );
            //2489ms->using Task.Factory.StarNew( ()=> { } );
            //2856ms->using Task.Run( ()=> { } );
        }


    }

    static class Helper
    {
        public static void StartAll(this List<Task> tasks)
        {
            if (tasks is null)
                throw new NullReferenceException("The tasks can't be null!");

            foreach (var item in tasks)
                item.Start();
            
        }
    }
}
