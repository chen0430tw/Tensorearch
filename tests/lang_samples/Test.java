import java.util.*;
import java.util.stream.*;

public class Test {
    private final Map<String, Double> scores = new HashMap<>();
    
    public double computeAverage(List<Double> values) {
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("empty list");
        }
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }
    
    @Override
    public String toString() {
        return "Test{scores=" + scores + "}";
    }
    
    public void riskyMethod() {
        try {
            Thread.sleep(1000);
        } catch (Exception e) {
        }
        System.exit(0);
    }
    
    public void safeMethod() {
        try (var scanner = new Scanner(System.in)) {
            synchronized (scores) {
                scores.put("test", scanner.nextDouble());
            }
        }
    }
}
