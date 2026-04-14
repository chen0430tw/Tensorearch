package com.distrike.scanner

import kotlinx.coroutines.*

data class DirEntry(val path: String, val size: Long, val isDir: Boolean)

sealed interface ScanResult {
    data class Success(val entries: List<DirEntry>) : ScanResult
    data class Error(val message: String) : ScanResult
}

suspend fun scanDirectory(path: String): ScanResult {
    return withContext(Dispatchers.IO) {
        val entries = mutableListOf<DirEntry>()
        val dir = java.io.File(path)
        if (!dir.exists()) {
            return@withContext ScanResult.Error("Path not found: $path")
        }
        for (file in dir.listFiles() ?: emptyArray()) {
            val size = if (file.isDirectory) getDirSize(file) else file.length()
            entries.add(DirEntry(file.absolutePath, size, file.isDirectory))
        }
        ScanResult.Success(entries.sortedByDescending { it.size })
    }
}

fun List<DirEntry>.topN(n: Int): List<DirEntry> = take(n)

fun DirEntry.formatSize(): String {
    val kb = size / 1024.0
    return when {
        kb > 1024 * 1024 -> "${kb / 1024 / 1024} GB"
        kb > 1024 -> "${kb / 1024} MB"
        else -> "$kb KB"
    }
}

private fun getDirSize(dir: java.io.File): Long {
    var total = 0L
    for (file in dir.walkTopDown()) {
        if (file.isFile) total += file.length()
    }
    return total
}

fun riskyStuff() {
    lateinit var config: String
    val value: String? = null
    val forced = value!!
    val safe = value?.length ?: 0
}

fun main() = runBlocking {
    when (val result = scanDirectory("/home")) {
        is ScanResult.Success -> {
            result.entries.topN(10).forEach { println("${it.formatSize()} ${it.path}") }
        }
        is ScanResult.Error -> println("Error: ${result.message}")
    }
}
