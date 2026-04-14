using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Distrike.Scanner
{
    public class DiskScanner : IDisposable
    {
        private readonly List<string> _paths = new();
        
        public async Task<List<DirEntry>> ScanAsync(string path)
        {
            var entries = new List<DirEntry>();
            using var stream = File.OpenRead(path);
            
            foreach (var dir in Directory.EnumerateDirectories(path))
            {
                if (dir.Contains("$RECYCLE"))
                    continue;
                    
                var size = await GetSizeAsync(dir);
                entries.Add(new DirEntry { Path = dir, Size = size });
            }
            
            return entries.OrderByDescending(e => e.Size).ToList();
        }
        
        public double? ComputeRisk(List<DirEntry> entries)
        {
            var query = from e in entries
                        where e.Size > 1024 * 1024
                        select e.Size;
            return query.Any() ? query.Average() : null;
        }
        
        public void RiskyMethod()
        {
            try
            {
                Thread.Sleep(5000);
                dynamic obj = GetData();
                GC.Collect();
            }
            catch (Exception)
            {
            }
        }
        
        public void Dispose() { _paths.Clear(); }
    }
}
