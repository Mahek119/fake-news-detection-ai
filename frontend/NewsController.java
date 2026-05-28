package com.example.capstone.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.client.SimpleClientHttpRequestFactory;

import java.util.HashMap;
import java.util.Map;

@Controller
public class NewsController {

    @GetMapping("/")
    public String home() {
        return "index";
    }

    @PostMapping("/news/check")
    public String checkNews(@RequestParam String text, Model model) {

        String url = "http://127.0.0.1:8000/predict";

        // Timeout config
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(3000);
        factory.setReadTimeout(5000);

        RestTemplate restTemplate = new RestTemplate(factory);

        Map<String, String> request = new HashMap<>();
        request.put("text", text);

        try {
           //Map response = restTemplate.postForObject(url, request, Map.class);
           Map<String, Object> response = restTemplate.postForObject(url, request, Map.class);

            if (response == null) {
                model.addAttribute("result", "ERROR");
                model.addAttribute("confidence", 0);
                return "index";
            }

            model.addAttribute("inputText", text);
            model.addAttribute("result", response.get("prediction"));
            model.addAttribute("confidence", response.get("confidence"));
            model.addAttribute("style", response.get("style_score"));
model.addAttribute("evidenceScore", response.get("evidence_score"));
model.addAttribute("evidence", response.get("evidence"));
model.addAttribute("explanation", response.get("explanation"));

        } catch (Exception e) {
            model.addAttribute("result", "ERROR");
            model.addAttribute("confidence", 0);
        }

        return "index";
    }
}

//package com.example.capstone.controller;

//import org.springframework.stereotype.Controller;
//import org.springframework.ui.Model;
//import org.springframework.web.bind.annotation.*;
//import org.springframework.web.client.RestTemplate;

//import java.util.HashMap;
//import java.util.Map;

//@Controller
//public class NewsController {

 //   @GetMapping("/")
 //   public String home() {
 //       return "index";
  //  }

 //   @PostMapping("/news/check")
  //  public String checkNews(@RequestParam String text, Model model) {

  //      String pythonApiUrl = "http://127.0.0.1:8000/predict";

  //      RestTemplate restTemplate = new RestTemplate();

   //     Map<String, String> request = new HashMap<>();
  //      request.put("text", text);

   //     Map response = restTemplate.postForObject(pythonApiUrl, request, Map.class);

   //     model.addAttribute("inputText", text);
   //     model.addAttribute("result", response.get("prediction"));
   //     model.addAttribute("style", response.get("style_score"));
   //     model.addAttribute("evidence", response.get("evidence_score"));

   //     return "index";
   // }
//}