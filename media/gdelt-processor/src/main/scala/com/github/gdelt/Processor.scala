package com.github.gdelt
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.io.File
import com.typesafe.scalalogging.slf4j.LazyLogging


class Processor extends LazyLogging {

  import spark.implicits._

  
  Processor()
  // GDELT GKG 2.0 Schema (27 fields, tab-delimited)
  val gkgSchema = StructType(Array(
    StructField("GKGRECORDID", StringType, nullable = true),
    StructField("DATE", StringType, nullable = true),
    StructField("SourceCollectionIdentifier", IntegerType, nullable = true),
    StructField("SourceCommonName", StringType, nullable = true),
    StructField("DocumentIdentifier", StringType, nullable = true),
    StructField("Counts", StringType, nullable = true),
    StructField("V2Counts", StringType, nullable = true),
    StructField("Themes", StringType, nullable = true),
    StructField("V2Themes", StringType, nullable = true),
    StructField("Locations", StringType, nullable = true),
    StructField("V2Locations", StringType, nullable = true),
    StructField("Persons", StringType, nullable = true),
    StructField("V2Persons", StringType, nullable = true),
    StructField("Organizations", StringType, nullable = true),
    StructField("V2Organizations", StringType, nullable = true),
    StructField("V2Tone", StringType, nullable = true),
    StructField("Dates", StringType, nullable = true),
    StructField("GCAM", StringType, nullable = true),
    StructField("SharingImage", StringType, nullable = true),
    StructField("RelatedImages", StringType, nullable = true),
    StructField("SocialImageEmbeds", StringType, nullable = true),
    StructField("SocialVideoEmbeds", StringType, nullable = true),
    StructField("Quotations", StringType, nullable = true),
    StructField("AllNames", StringType, nullable = true),
    StructField("Amounts", StringType, nullable = true),
    StructField("TranslationInfo", StringType, nullable = true),
    StructField("Extras", StringType, nullable = true)
  ))
  
  def loadGKGData(dirPath: String = "../media/media_data"): DataFrame = {
    
    val csvfiles = new File(dirPath)
      .listFiles()
      .filter(_.getName.endsWith(".csv"))
      .map(_.getAbsolutePath)
      .toList
    
    if(csvFiles.isEmpty()) {
      throw new RuntimeException("No Media Files Downloaded by crawler")
      
    }
    
    val df = spark.read
      .option("delimiter", ";")
      .option("head","false")
      .option("quote","")
      .option("escape","\\")
      .option("mode","PERMISSIVE")
      .schema(gkgSchema)
      .csv(csvfiles: _*)
    
    logger.info("Df successfully loaded from disk.")
    
    df
  }
  
  def parseLocations(location: String) -> {
    if(string == null) {
      Array.empty[Map[String,String]]
    } else {
      string.split("")
    }
  }
  
  
}
