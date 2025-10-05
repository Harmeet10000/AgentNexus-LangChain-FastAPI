"""Crawl repository implementation."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import Depends

from core.database.mongodb import get_database
from core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class CrawlRepository:
    """Repository for crawl data operations."""
    
    def __init__(self, db):
        """Initialize crawl repository."""
        self.db = db
        self.crawl_jobs = db.crawl_jobs
        self.crawl_results = db.crawl_results
    
    async def save_crawl_job(self, job_data: Dict[str, Any]) -> str:
        """Save crawl job to database."""
        try:
            job_doc = {
                **job_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "pages_crawled": 0
            }
            
            result = await self.crawl_jobs.insert_one(job_doc)
            
            logger.info(f"Saved crawl job with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to save crawl job", error=str(e))
            raise
    
    async def get_crawl_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get crawl job by ID."""
        try:
            job = await self.crawl_jobs.find_one({"job_id": job_id})
            
            if job:
                job["_id"] = str(job["_id"])
                logger.info(f"Retrieved crawl job: {job_id}")
            
            return job
            
        except Exception as e:
            logger.error("Failed to get crawl job", error=str(e))
            raise
    
    async def update_crawl_status(
        self, 
        job_id: str, 
        status: str,
        error_message: Optional[str] = None,
        results_summary: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update crawl job status."""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if error_message:
                update_data["error_message"] = error_message
            
            if results_summary:
                update_data["results_summary"] = results_summary
                update_data["pages_crawled"] = results_summary.get("total_pages", 0)
            
            result = await self.crawl_jobs.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated crawl job status: {job_id} -> {status}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error("Failed to update crawl status", error=str(e))
            raise
    
    async def save_crawl_results(
        self, 
        job_id: str, 
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Save crawl results to database."""
        try:
            result_docs = []
            for i, result in enumerate(results):
                result_doc = {
                    "job_id": job_id,
                    "result_index": i,
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "links": result.get("links", []),
                    "images": result.get("images", []),
                    "metadata": result.get("metadata", {}),
                    "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
                    "created_at": datetime.utcnow()
                }
                result_docs.append(result_doc)
            
            if result_docs:
                result = await self.crawl_results.insert_many(result_docs)
                result_ids = [str(id) for id in result.inserted_ids]
                logger.info(f"Saved {len(result_ids)} crawl results for job {job_id}")
                return result_ids
            
            return []
            
        except Exception as e:
            logger.error("Failed to save crawl results", error=str(e))
            raise
    
    async def get_crawl_results(
        self, 
        job_id: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get crawl results for a job."""
        try:
            cursor = self.crawl_results.find(
                {"job_id": job_id}
            ).sort("result_index", 1).limit(limit)
            
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format for response
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result["url"],
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "links": result.get("links"),
                    "images": result.get("images"),
                    "metadata": result.get("metadata"),
                    "timestamp": result.get("timestamp")
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Retrieved {len(formatted_results)} crawl results for job {job_id}")
            return formatted_results
            
        except Exception as e:
            logger.error("Failed to get crawl results", error=str(e))
            raise
    
    async def list_crawl_jobs(
        self, 
        status: Optional[str] = None,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List crawl jobs with optional filtering."""
        try:
            query = {}
            if status:
                query["status"] = status
            
            cursor = self.crawl_jobs.find(query).skip(skip).limit(limit).sort("created_at", -1)
            jobs = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for job in jobs:
                job["_id"] = str(job["_id"])
            
            logger.info(f"Retrieved {len(jobs)} crawl jobs")
            return jobs
            
        except Exception as e:
            logger.error("Failed to list crawl jobs", error=str(e))
            raise
    
    async def delete_crawl_job(self, job_id: str) -> bool:
        """Delete crawl job and its results."""
        try:
            # Delete results first
            await self.crawl_results.delete_many({"job_id": job_id})
            
            # Delete job
            result = await self.crawl_jobs.delete_one({"job_id": job_id})
            
            logger.info(f"Deleted crawl job: {job_id}")
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error("Failed to delete crawl job", error=str(e))
            raise
    
    async def get_crawl_stats(self) -> Dict[str, Any]:
        """Get crawl statistics."""
        try:
            total_jobs = await self.crawl_jobs.count_documents({})
            total_results = await self.crawl_results.count_documents({})
            
            # Get jobs by status
            pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_stats = await self.crawl_jobs.aggregate(pipeline).to_list(length=None)
            
            # Get recent jobs
            recent_jobs = await self.crawl_jobs.find({}).sort("created_at", -1).limit(10).to_list(length=10)
            
            stats = {
                "total_jobs": total_jobs,
                "total_results": total_results,
                "by_status": {stat["_id"]: stat["count"] for stat in status_stats},
                "recent_jobs": [
                    {
                        "job_id": job["job_id"],
                        "url": job["url"],
                        "status": job["status"],
                        "created_at": job["created_at"].isoformat()
                    }
                    for job in recent_jobs
                ]
            }
            
            logger.info("Retrieved crawl statistics")
            return stats
            
        except Exception as e:
            logger.error("Failed to get crawl statistics", error=str(e))
            raise


def get_crawl_repository(db = Depends(get_database)) -> CrawlRepository:
    """Dependency to get crawl repository."""
    return CrawlRepository(db)