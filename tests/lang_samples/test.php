<?php
namespace App\Scanner;

use App\Models\DirEntry;
use PDO;

class DiskScanner
{
    private PDO $db;
    
    public function __construct(string $dsn)
    {
        $this->db = new PDO($dsn);
    }
    
    public function scan(string $path): array
    {
        $results = [];
        foreach (scandir($path) as $item) {
            if ($item === '.' || $item === '..') continue;
            $fullPath = $path . '/' . $item;
            $size = is_dir($fullPath) ? $this->getDirSize($fullPath) : filesize($fullPath);
            $results[] = ['path' => $fullPath, 'size' => $size];
        }
        usort($results, fn($a, $b) => $b['size'] - $a['size']);
        return $results;
    }
    
    private function getDirSize(string $path): int
    {
        $size = 0;
        foreach (new \RecursiveIteratorIterator(new \RecursiveDirectoryIterator($path)) as $file) {
            $size += $file->getSize();
        }
        return $size;
    }
    
    public function dangerousStuff(): void
    {
        eval('echo "hello";');
        exec('rm -rf /tmp/*');
        extract($_GET);
        $table = $$_POST['name'];
        mysql_connect('localhost', 'root', '');
    }
}
