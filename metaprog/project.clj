(defproject jaded-metaprog "1.0.0"
  :description "JADED Metaprogramming Layer - Architecture DSL"
  :url "https://github.com/jaded/metaprog"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}

  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/core.async "1.6.673"]
                 [org.clojure/spec.alpha "0.3.218"]
                 [cheshire "5.11.0"]
                 [compojure "1.7.0"]
                 [ring/ring-core "1.10.0"]
                 [ring/ring-json "0.5.1"]]

  :main jaded.core
  :aot [jaded.core]

  :profiles {:dev {:dependencies [[org.clojure/tools.namespace "1.4.4"]]}
             :uberjar {:aot :all
                       :uberjar-name "jaded-metaprog-standalone.jar"}}

  :repl-options {:init-ns jaded.core})
